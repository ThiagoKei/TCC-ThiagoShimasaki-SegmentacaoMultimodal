"""Modelo de landmarks por dente (decisões: pontos da coroa + eixo longo IE).

Entrada: recorte do dente (B, 512, 7: xyz e normal no referencial local + flag dente) + condição do tipo
(posição 1–8 one-hot, grupo one-hot, arcada) + (opcional) token 2D do mesmo FDI e tip_2d.

Codificador compartilhado: DGCNN pequeno (3 EdgeConv fatorados, grafo geométrico fixo, k = 16) — o mesmo backbone da 2ª colocada do 3DTeethLand
(ChohoTech) e do Kim 2026 (eixo pela coroa).

Cabeça PONTOS (treinada no 3DTeethLand): por ponto do recorte, para cada um dos 6 canais (Mesial, Distal,
  FacialPoint, InnerPoint, OuterPoint, Cusp) um logit de "estou perto" e um deslocamento até o landmark
  (heatmap + deslocamento, como a ChohoTech). Cúspide = canal único "perto da cúspide mais próxima"; o número
  variável de cúspides sai por agrupamento dos votos (ver extrair_pontos).
Cabeça EIXO (treinada nos casos do planejador): feature global (+ condição) → MLP → vetor unitário IE.
  Com --com-2d: fusão TARDIA com o token 2D do mesmo FDI (a correspondência já está resolvida pelo FDI, então não
  precisa de atenção cruzada) + tip_2d explícito + flag tem_2d.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

N_CANAIS = 6            # Mesial, Distal, FacialPoint, InnerPoint, OuterPoint, Cusp
CANAL_DENTE = N_CANAIS  # 7º logit por ponto = "este ponto é do dente" (refino do contorno)
D_COND = 8 + 4 + 1      # posição one-hot, grupo one-hot, arcada (1 = superior)


def knn(x, k):
    """x (B, C, N) → índices (B, N, k) dos k vizinhos mais próximos no espaço de x."""
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = (x ** 2).sum(1, keepdim=True)
    dist = -xx - inner - xx.transpose(2, 1)
    return dist.topk(k=k, dim=-1)[1]


class EdgeConv(nn.Module):
    """EdgeConv FATORADO e agregado antes da normalização (variante leve do DGCNN, para CPU):
    W·[x_j − x_i, x_i] = W₁·x_j + (W₂ − W₁)·x_i  →  h_i = max_k(W₁·x_j) + (W₂ − W₁)·x_i,  depois BN + LeakyReLU.
    Não monta o tensor (B, 2C, N, k) nem normaliza N × k valores; o grafo (idx) é o k-NN geométrico, fixo."""

    def __init__(self, c_in, c_out):
        super().__init__()
        self.viz = nn.Conv1d(c_in, c_out, 1, bias=False)          # W₁
        self.cen = nn.Conv1d(c_in, c_out, 1, bias=False)          # W₂ − W₁
        self.bn = nn.BatchNorm1d(c_out)

    def forward(self, x, idx):
        B, _, N = x.shape
        a = self.viz(x)                                            # (B, C_out, N)
        C, k = a.shape[1], idx.shape[-1]
        base = torch.arange(B, device=x.device).view(-1, 1, 1) * N
        g = a.transpose(1, 2).reshape(B * N, C)[(idx + base).view(-1)].view(B, N, k, C).max(dim=2)[0]
        return F.leaky_relu(self.bn(g.transpose(1, 2) + self.cen(x)), 0.2)


class ModeloLandmarks(nn.Module):
    def __init__(self, com_2d=False, k=16, emb=256, d_tok=1280, d_2d=64, p_drop=0.3):
        super().__init__()
        self.k, self.com_2d = k, com_2d
        self.e1, self.e2, self.e3 = EdgeConv(7, 64), EdgeConv(64, 64), EdgeConv(64, 128)
        self.cond = nn.Sequential(nn.Linear(D_COND, 64), nn.ReLU())
        self.proj = nn.Sequential(nn.Conv1d(256 + 64, emb, 1, bias=False), nn.BatchNorm1d(emb), nn.LeakyReLU(0.2))
        # cabeça dos pontos: por ponto, [features locais 256, global 2·emb → 256, condição 64]
        self.glob_red = nn.Linear(2 * emb, 256)
        self.pontos = nn.Sequential(
            nn.Conv1d(256 + 256 + 64, 128, 1, bias=False), nn.BatchNorm1d(128), nn.LeakyReLU(0.2),
            nn.Conv1d(128, 64, 1, bias=False), nn.BatchNorm1d(64), nn.LeakyReLU(0.2),
            nn.Conv1d(64, N_CANAIS * 4 + 1, 1))                   # 6 logits + 6 deslocamentos (x, y, z) + dente
        # cabeça do eixo
        d_in = 2 * emb + 64
        if com_2d:
            self.tok = nn.Sequential(nn.LayerNorm(d_tok), nn.Linear(d_tok, d_2d), nn.ReLU())
            d_in += d_2d + 2                                       # + tip_2d (/30) + tem_2d
        self.eixo = nn.Sequential(nn.Linear(d_in, 256), nn.ReLU(), nn.Dropout(p_drop),
                                  nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 3))

    def forward(self, pts, cond, token=None, tip2d=None, tem2d=None):
        """pts (B, N, 7); cond (B, 13). Devolve logits (B, N, 7: os 6 canais dos pontos + CANAL_DENTE),
        desloc (B, N, 6, 3), ie (B, 3) unitário."""
        B, N, _ = pts.shape
        x = pts.transpose(1, 2)
        idx = knn(x[:, :3], self.k)                                # grafo geométrico (xyz), o mesmo nas 3 camadas
        x1 = self.e1(x, idx)
        x2 = self.e2(x1, idx)
        x3 = self.e3(x2, idx)
        c = self.cond(cond)                                        # (B, 64)
        loc = torch.cat([x1, x2, x3], 1)                           # (B, 256, N)
        f = self.proj(torch.cat([loc, c[:, :, None].expand(-1, -1, N)], 1))
        g = torch.cat([f.max(-1)[0], f.mean(-1)], 1)               # (B, 2·emb)
        gp = self.glob_red(g)[:, :, None].expand(-1, -1, N)
        o = self.pontos(torch.cat([loc, gp, c[:, :, None].expand(-1, -1, N)], 1)).transpose(1, 2)
        logits = torch.cat([o[..., :N_CANAIS], o[..., -1:]], -1)
        desloc = o[..., N_CANAIS:N_CANAIS * 4].reshape(B, N, N_CANAIS, 3)
        h = [g, c]
        if self.com_2d:
            if token is None:                                      # sem 2D (ex.: 3DTeethLand na avaliação)
                token, tip2d, tem2d = pts.new_zeros(B, self.tok[0].normalized_shape[0]), pts.new_zeros(B), \
                    pts.new_zeros(B)
            h += [self.tok(token.float()) * tem2d[:, None], (tip2d / 30.0)[:, None] * tem2d[:, None], tem2d[:, None]]
        ie = F.normalize(self.eixo(torch.cat(h, 1)), dim=1)
        return logits, desloc, ie


# ── alvos e extração ──────────────────────────────────────────────────────────────────────────────────────────────
def alvos_pontos(xyz, kp, kp_mask, sigma):
    """xyz (B, N, 3); kp (B, 11, 3) [5 únicos + 6 cúspides]; kp_mask (B, 11).
    Devolve calor (B, N, 6) ∈ [0, 1], desloc (B, N, 6, 3) até o landmark (cúspide: a mais próxima), valido (B, 6)."""
    B, N, _ = xyz.shape
    uni, mu = kp[:, :5], kp_mask[:, :5]
    d_uni = uni[:, None] - xyz[:, :, None]                         # (B, N, 5, 3)
    cus, mc = kp[:, 5:], kp_mask[:, 5:]
    d_cus = cus[:, None] - xyz[:, :, None]                         # (B, N, 6, 3)
    dist_c = d_cus.norm(dim=-1).masked_fill(~mc[:, None], 1e6)
    j = dist_c.argmin(-1)                                          # cúspide mais próxima de cada ponto
    d_c = torch.gather(d_cus, 2, j[..., None, None].expand(-1, -1, 1, 3))[:, :, 0]
    desloc = torch.cat([d_uni, d_c[:, :, None]], 2)                # (B, N, 6, 3)
    calor = torch.exp(-desloc.norm(dim=-1) ** 2 / (2 * sigma ** 2))
    valido = torch.cat([mu, mc.any(1, keepdim=True)], 1)
    return calor, desloc, valido


def extrair_pontos(xyz, logits, desloc, top=16, limiar_cusp=0.3, eps=1.0, score_cusp="soma"):
    """Um recorte (numpy): xyz (N, 3), logits (N, 6), desloc (N, 6, 3).
    Únicos: média ponderada dos votos dos `top` pontos de maior calor. Cúspides: votos com calor > limiar,
    agrupados (DBSCAN ponderado, eps mm); score = soma do calor do grupo (padrão) ou calor máximo do grupo
    (score_cusp="max", escolhido no val do 3DTeethLand). Devolve (unicos (5, 3), scores (5,),
    cúspides [(xyz, score)])."""
    from sklearn.cluster import DBSCAN
    p = 1 / (1 + np.exp(-logits))
    votos = xyz[:, None] + desloc
    uni, sc = np.zeros((5, 3)), np.zeros(5)
    for c in range(5):
        i = np.argsort(-p[:, c])[:top]
        w = p[i, c]
        uni[c], sc[c] = (votos[i, c] * w[:, None]).sum(0) / max(w.sum(), 1e-6), float(p[i[0], c])
    cusp = []
    m = p[:, 5] > limiar_cusp
    if m.sum() >= 3:
        v, w = votos[m, 5], p[m, 5]
        lab = DBSCAN(eps=eps, min_samples=3).fit_predict(v, sample_weight=w)
        for g in set(lab) - {-1}:
            q = lab == g
            cusp.append(((v[q] * w[q, None]).sum(0) / w[q].sum(),
                         float(w[q].max() if score_cusp == "max" else w[q].sum())))
    return uni, sc, sorted(cusp, key=lambda t: -t[1])
