"""MeshSegNet + fusão 2D → 3D (atenção cruzada por dente).

Herda o MeshSegNet público (00_comum/meshsegnet.py) com os MESMOS nomes de camada, então
os pesos pré-treinados carregam direto (strict=False só pelas camadas novas).

Onde entra o 2D: depois do GLM-2 (x_glm2, 512 canais por célula) e antes do max pooling global, para
que o vetor global e a fusão densa já vejam a informação da panorâmica:

    x_glm2' = x_glm2 + tanh(portão) · Atenção(células → tokens dos dentes da panorâmica)

- Consultas: x_glm2 projetado + codificação de Fourier da coordenada do arco (0 = lado direito).
- Chaves/valores: um token por dente do Mask R-CNN v2 (feat_box, feat_mask, probs, score, área) +
  Fourier de pos_norm (x, y), mais um TOKEN NULO aprendido, sempre presente. Uma célula que não acha
  dente correspondente (gengiva, dente que o 2D perdeu, panorâmica ausente) atende ao nulo.
- Sem calibração: nenhuma projeção 3D → 2D; a correspondência é aprendida (soft association do
  TransFusion). A posição ao longo do arco nas duas modalidades é o que ancora a atenção.
- Portão começa em 0: no início o modelo é EXATAMENTE o MeshSegNet pré-treinado; o 2D só entra se ajudar
  (inicialização "zero" do Flamingo / ControlNet). Isso protege os ~390 casos de treino.
- Robustez: dropout de modalidade (todos os tokens somem com prob. p_sem_2d) e de token (cada dente
  some com prob. p_token) no treino, para o modelo não depender de um 2D perfeito.

Variantes (--fusao): "nenhuma" (MeshSegNet retreinado nos mesmos casos — a comparação obrigatória),
"film" (condicionamento global) e "atencao". A atenção é LIVRE por
padrão; com prior=True (--prior no treino) é a atenção GUIADA, o modelo proposto: os logits de
cada cabeça recebem −λ_h [(f̂(s_i) − x_j)² − min_k(f̂(s_i) − x_k)²] / 2σ², com f̂ do registro_fusao.py, λ_h aprendido e dropout do prior.

A atenção devolve os pesos (B, N, 1+T): dá para ver, por célula, qual dente da panorâmica ela usou —
material para a discussão (e para conferir se a correspondência aprendida faz sentido anatômico).
"""
import copy
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path[:0] = [str(p) for p in sorted(Path(__file__).resolve().parents[1].glob("0*_*"))]  # pastas do repositório
from meshsegnet import MeshSegNet  # noqa: E402


class Fourier(nn.Module):
    """Codificação de Fourier de k coordenadas em [0, 1] → (…, k·2·n_freq)."""

    def __init__(self, k, n_freq=8):
        super().__init__()
        self.register_buffer("freq", (2.0 ** torch.arange(n_freq)) * math.pi, persistent=False)
        self.dim = k * 2 * n_freq

    def forward(self, x):                       # x: (…, k)
        ang = x.unsqueeze(-1) * self.freq       # (…, k, n_freq)
        return torch.cat([ang.sin(), ang.cos()], -1).flatten(-2)


class TokensDentes(nn.Module):
    """Token por dente da panorâmica: ramos separados por tipo de feature (escalas muito diferentes)."""

    def __init__(self, d=256):
        super().__init__()
        self.box = nn.Sequential(nn.LayerNorm(1024), nn.Linear(1024, d))
        self.mask = nn.Sequential(nn.LayerNorm(256), nn.Linear(256, d // 2))
        self.meta = nn.Linear(32 + 2, d // 2)              # probs FDI + score + área
        self.pos = Fourier(2)
        self.junta = nn.Sequential(nn.Linear(d + d // 2 + d // 2 + self.pos.dim, d), nn.GELU(), nn.Linear(d, d))
        self.nulo = nn.Parameter(torch.zeros(1, 1, d))
        nn.init.normal_(self.nulo, std=0.02)

    def forward(self, tokens, valido):
        """tokens (B, T, 1316), valido (B, T) → chaves (B, 1+T, d) e máscara de padding (B, 1+T)."""
        box, mask, meta, pos = tokens[..., :1024], tokens[..., 1024:1280], tokens[..., 1280:1314], tokens[..., 1314:]
        t = self.junta(torch.cat([self.box(box), self.mask(mask), self.meta(meta), self.pos(pos)], -1))
        B = t.shape[0]
        chaves = torch.cat([self.nulo.expand(B, -1, -1), t], 1)
        ignorar = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=valido.device), ~valido], 1)
        return chaves, ignorar


class AtencaoCruzada(nn.Module):
    def __init__(self, c=512, d=256, cabecas=8, prior=False, portao_inicial=0.0, portao_celula=False,
                 portao_disc=2.0, portao_b0=-3.0, portao_x=False):
        super().__init__()
        self.pos = Fourier(1)
        self.q = nn.Sequential(nn.LayerNorm(c + self.pos.dim), nn.Linear(c + self.pos.dim, d))
        self.attn = nn.MultiheadAttention(d, cabecas, batch_first=True)
        self.saida = nn.Linear(d, c)
        # tanh(0) = 0 → começa como o MeshSegNet; > 0 abre a porta do 2D desde o início
        self.portao = nn.Parameter(torch.full((1,), math.atanh(portao_inicial)))
        self.cabecas, self.prior = cabecas, prior
        # Portão POR CÉLULA no lugar do escalar. g_i = σ(b + a·disc_i + c·nulo_dente_i + v·x_i), com
        #   disc_i       massa da atenção em tokens cujo número ≠ o do rascunho 3D na célula (o 2D discorda dele ali),
        #   nulo_dente_i peso no token nulo onde o rascunho 3D diz dente (a panorâmica não mostra dente ali).
        # b = −3 → g ≈ 0,05: quase fechado, mas com gradiente (com 0 exato a atenção não aprende, rodada 1);
        # a começa em portao_disc (> 0): onde o 2D discorda do rascunho 3D a célula nasce mais aberta (σ(−1) ≈ 0,27).
        # O rascunho_conf não serve de sinal: 98% das células com confiança 1,0, AUC 0,56 para achar o erro.
        # O termo v·x é opcional (portao_x): no teste em CPU, com x cru (escala grande) ele levou g a 1,0 em TODAS as
        # células em 10 passos e matou a seletividade. Por padrão o portão só vê os dois sinais de discordância.
        self.portao_celula, self.portao_x = portao_celula, portao_x
        if portao_celula:
            self.g_b = nn.Parameter(torch.tensor(float(portao_b0)))
            self.g_sinais = nn.Parameter(torch.tensor([float(portao_disc), 0.0]))
            if portao_x:
                self.g_x = nn.Sequential(nn.LayerNorm(c), nn.Linear(c, 1))
                nn.init.zeros_(self.g_x[1].weight)
                nn.init.zeros_(self.g_x[1].bias)
            self.zerar_g()
        if prior:                                          # λ_h = softplus(ρ_h), começa em 1
            self.rho = nn.Parameter(torch.full((cabecas,), math.log(math.e - 1)))

    def zerar_g(self):
        """Acumuladores do portão por célula (média de g, e g nas células com discordância), para o log."""
        self.soma_g, self.n_g, self.soma_g_disc, self.n_g_disc = 0.0, 0, 0.0, 0

    def vies(self, centro, x_tok, sigma, tem, ignorar):
        """Prior geométrico do registro (atenção GUIADA), somado aos logits de cada cabeça:
            b_hij = −λ_h · [(f̂(s_i) − x_j)² − min_k (f̂(s_i) − x_k)²] / 2σ²   (dentes j ≥ 1)
        O dente mais perto de f̂(s_i) fica com 0, como o token nulo: o prior só escolhe QUAL dente; "dente ou
        nada" (gengiva, dente perdido) continua com o conteúdo. Sem o "− min", λ alto empurrava tudo para o
        nulo. Token nulo, token sem x e amostra sem prior: 0.
        centro (B, N) = f̂(s_i); x_tok (B, T) = x da coroa do token j; sigma, tem (B,).
        O padding entra como −inf aqui mesmo (uma máscara float só, sem misturar com a booleana)."""
        d2 = ((centro.unsqueeze(-1) - torch.nan_to_num(x_tok).unsqueeze(1)) / sigma.view(-1, 1, 1)).square()
        fora = ignorar[:, 1:].unsqueeze(1) | torch.isnan(x_tok).unsqueeze(1)          # (B, 1, T)
        mais_perto = d2.masked_fill(fora, float("inf")).amin(-1, keepdim=True)
        mais_perto = torch.where(torch.isinf(mais_perto), torch.zeros_like(mais_perto), mais_perto)
        b = (-0.5 * (d2 - mais_perto)).masked_fill(fora, 0.0) * tem.view(-1, 1, 1).float()
        b = torch.cat([torch.zeros_like(b[..., :1]), b], -1)                         # token nulo
        lam = F.softplus(self.rho).view(1, -1, 1, 1)
        m = lam * b.unsqueeze(1) + torch.zeros_like(ignorar, dtype=b.dtype).masked_fill(
            ignorar, float("-inf")).view(ignorar.shape[0], 1, 1, -1)
        return m.flatten(0, 1)                                                       # (B·H, N, 1+T)

    def forward(self, x, arco, chaves, ignorar, prior=None, cls_tok=None, rascunho_cel=None):
        """x (B, 512, N), arco (B, N) → x fundido (B, 512, N) e pesos de atenção (B, N, 1+T).
        prior = (centro, x_tok, sigma, tem) do registro; None ou self.prior False → atenção livre.
        cls_tok (B, T) classe 0..16 de cada token (−1 fora da arcada) e rascunho_cel (B, N) a predição crua do rascunho 3D:
        só o portão por célula usa (sem eles, disc = nulo_dente = 0 e g fica no viés)."""
        q = self.q(torch.cat([x.transpose(1, 2), self.pos(arco.unsqueeze(-1))], -1))
        if self.prior and prior is not None:
            h, pesos = self.attn(q, chaves, chaves, attn_mask=self.vies(*prior, ignorar), need_weights=True)
        else:
            h, pesos = self.attn(q, chaves, chaves, key_padding_mask=ignorar, need_weights=True)
        if not self.portao_celula:
            return x + torch.tanh(self.portao) * self.saida(h).transpose(1, 2), pesos
        if cls_tok is not None and rascunho_cel is not None:
            difere = (cls_tok.unsqueeze(1) != rascunho_cel.unsqueeze(-1)).to(pesos.dtype)          # (B, N, T)
            disc = (pesos[..., 1:] * difere).sum(-1)
            nulo_dente = pesos[..., 0] * (rascunho_cel > 0).to(pesos.dtype)
        else:
            disc = nulo_dente = torch.zeros_like(pesos[..., 0])
        logit = self.g_b + self.g_sinais[0] * disc + self.g_sinais[1] * nulo_dente
        if self.portao_x:
            logit = logit + self.g_x(x.transpose(1, 2)).squeeze(-1)
        g = torch.sigmoid(logit)                                                             # (B, N)
        with torch.no_grad():
            self.soma_g += float(g.sum())
            self.n_g += g.numel()
            m = disc > 0.5
            self.soma_g_disc += float(g[m].sum())
            self.n_g_disc += int(m.sum())
        return x + g.unsqueeze(1) * self.saida(h).transpose(1, 2), pesos


class FiLM(nn.Module):
    """Linha de base: γ, β por canal a partir do fpn_global + média dos tokens (sem correspondência)."""

    def __init__(self, c=512, d=256):
        super().__init__()
        self.rede = nn.Sequential(nn.LayerNorm(1280 + d), nn.Linear(1280 + d, d), nn.GELU(), nn.Linear(d, 2 * c))
        nn.init.zeros_(self.rede[-1].weight)
        nn.init.zeros_(self.rede[-1].bias)                 # γ = β = 0 → começa como o MeshSegNet

    def forward(self, x, glob, chaves, ignorar):
        validos = (~ignorar[:, 1:]).unsqueeze(-1).float()
        media = (chaves[:, 1:] * validos).sum(1) / validos.sum(1).clamp(min=1)
        gama, beta = self.rede(torch.cat([glob, media], -1)).chunk(2, -1)
        return x * (1 + gama.unsqueeze(-1)) + beta.unsqueeze(-1)


CABECA = ["mlp3_conv1", "mlp3_bn1_1", "mlp3_conv2", "mlp3_bn1_2", "mlp3_conv3", "mlp3_bn2_1", "mlp3_conv4",
          "mlp3_bn2_2", "output_conv"]
NOVOS = {"tokens", "cruzada", "film", "cab_fus", "delta"}      # camadas que não existem no MeshSegNet pré-treinado


class MeshSegNetFusao(MeshSegNet):
    def __init__(self, num_classes=17, num_channels=15, fusao="atencao", d=256, cabecas=8,
                 p_sem_2d=0.2, p_token=0.1, usar_probs=True, prior=False, p_sem_prior=0.1, portao_inicial=0.0,
                 portao_celula=False, portao_disc=2.0, portao_b0=-3.0, portao_x=False, assimetrica=False,
                 residual=False):
        super().__init__(num_classes=num_classes, num_channels=num_channels)
        # Fusão RESIDUAL: P_3D = softmax(z_3D) e P_fus = softmax(z_3D + Δ), com Δ = cabeça leve
        # (por célula) sobre [features 3D, contribuição do 2D] e a última camada ZERADA → no início P_fus = P_3D
        # exatamente, e o 2D só entra como correção. Sem stop-gradient: a L_fus continua chegando ao tronco (o
        # "professor" do Z4); a L_3D explícita mantém o caminho só 3D bom em todo lote. O portão por célula compara
        # o token com a P_3D do próprio modelo (como a assimétrica). sem_2d=True devolve a P_3D.
        if assimetrica and residual:
            raise ValueError("--assimetrica e --residual são excludentes")
        self.residual = residual and fusao == "atencao"
        if self.residual:
            self.delta = nn.Sequential(nn.Conv1d(1024, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
                                       nn.Conv1d(128, num_classes, 1))
            nn.init.zeros_(self.delta[-1].weight)
            nn.init.zeros_(self.delta[-1].bias)
        # Perda assimétrica ("3D decide, 2D corrige"): a cabeça roda duas vezes, sobre as features 3D
        # (P_3D, guardada em ultimo_p3d) e sobre as fundidas (P_fus, a saída). No treino a fusão recebe as features do
        # tronco SEM gradiente, então o tronco só aprende com a perda da P_3D e não tem como terceirizar a numeração.
        # O portão por célula compara o token com a predição 3D do próprio modelo (não com a de um modelo externo).
        # sem_2d=True devolve a P_3D.
        self.assimetrica = assimetrica and fusao != "nenhuma"
        if self.assimetrica:   # cabeça própria da fusão: a do 3D fica idêntica a um modelo só 3D (só a L_3D a treina)
            self.cab_fus = nn.ModuleDict({k: copy.deepcopy(getattr(self, k)) for k in CABECA})
        self.usar_probs = usar_probs       # False: ablação sem as probs FDI do 2D (o risco de só copiar o número)
        if fusao not in {"nenhuma", "film", "atencao"}:
            raise ValueError(f"fusao inválida: {fusao}")
        self.fusao, self.p_sem_2d, self.p_token = fusao, p_sem_2d, p_token
        self.prior, self.p_sem_prior = prior and fusao == "atencao", p_sem_prior
        if fusao != "nenhuma":
            self.tokens = TokensDentes(d)
            self.cruzada = AtencaoCruzada(512, d, cabecas, prior=self.prior, portao_inicial=portao_inicial,
                                          portao_celula=portao_celula, portao_disc=portao_disc,
                                          portao_b0=portao_b0, portao_x=portao_x) if fusao == "atencao" else None
            self.film = FiLM(512, d) if fusao == "film" else None

    def parametros_novos(self):
        """Só as camadas da fusão (para treiná-las primeiro, com o MeshSegNet congelado)."""
        return [p for n, p in self.named_parameters() if n.split(".")[0] in NOVOS]

    def _dropout_2d(self, valido):
        if not self.training:
            return valido
        B, T = valido.shape
        some_tudo = torch.rand(B, 1, device=valido.device) < self.p_sem_2d
        some_dente = torch.rand(B, T, device=valido.device) < self.p_token
        return valido & ~some_tudo & ~some_dente

    def forward(self, x, a_s, a_l, arco=None, tokens=None, valido=None, glob=None, sem_2d=False, prior=None,
                cls_tok=None, rascunho_cel=None):
        """Mesmo contrato do MeshSegNet (x, a_s, a_l) + o 2D. sem_2d=True: avaliação com o 2D desligado.
        prior = (centro, x_tok, sigma, tem) do registro (só com prior=True no construtor).
        Devolve (prob (B, N, C), pesos de atenção ou None)."""
        if self.prior and prior is not None and self.training and self.p_sem_prior:
            centro, x_tok, sigma, tem = prior               # dropout do prior: às vezes a atenção fica livre
            prior = (centro, x_tok, sigma, tem & (torch.rand_like(sigma) >= self.p_sem_prior))
        n_pts = x.size()[2]
        x = F.relu(self.mlp1_bn1(self.mlp1_conv1(x)))
        x = F.relu(self.mlp1_bn2(self.mlp1_conv2(x)))
        trans_feat = self.fstn(x)
        x_ftm = torch.bmm(x.transpose(2, 1), trans_feat)
        sap = torch.bmm(a_s, x_ftm).transpose(2, 1)
        x_ftm = x_ftm.transpose(2, 1)
        x = F.relu(self.glm1_bn1_1(self.glm1_conv1_1(x_ftm)))
        x = torch.cat([x, F.relu(self.glm1_bn1_2(self.glm1_conv1_2(sap)))], dim=1)
        x = F.relu(self.glm1_bn2(self.glm1_conv2(x)))
        x = F.relu(self.mlp2_bn1(self.mlp2_conv1(x)))
        x = F.relu(self.mlp2_bn2(self.mlp2_conv2(x)))
        x_mlp2 = F.relu(self.mlp2_bn3(self.mlp2_conv3(x)))
        if self.with_dropout:
            x_mlp2 = self.dropout(x_mlp2)
        xt = x_mlp2.transpose(2, 1)
        sap_1, sap_2 = torch.bmm(a_s, xt).transpose(2, 1), torch.bmm(a_l, xt).transpose(2, 1)
        x = torch.cat([F.relu(self.glm2_bn1_1(self.glm2_conv1_1(x_mlp2))),
                       F.relu(self.glm2_bn1_2(self.glm2_conv1_2(sap_1))),
                       F.relu(self.glm2_bn1_3(self.glm2_conv1_3(sap_2)))], dim=1)
        x_glm2 = F.relu(self.glm2_bn2(self.glm2_conv2(x)))

        # ── fusão 2D → 3D ──────────────────────────────────────────────────
        pesos = None
        if self.assimetrica or self.residual:
            z3 = self._cabeca(x_glm2, x_ftm, x_mlp2, n_pts, logits=True)
            p3d = torch.softmax(z3, dim=-1)
            self.ultimo_p3d = p3d
            if sem_2d:
                return p3d, None
            rascunho_cel = p3d.argmax(-1).detach()
            if self.assimetrica and self.training:
                x_glm2, x_ftm, x_mlp2 = x_glm2.detach(), x_ftm.detach(), x_mlp2.detach()
        x_glm2_3d = x_glm2
        if self.fusao != "nenhuma":
            v = torch.zeros_like(valido) if sem_2d else self._dropout_2d(valido)
            if not self.usar_probs:
                tokens = torch.cat([tokens[..., :1280], torch.zeros_like(tokens[..., 1280:1312]), tokens[..., 1312:]], -1)
            chaves, ignorar = self.tokens(tokens, v)
            self.ultimo_ignorar = ignorar      # (B, 1+T), para a supervisão da atenção no treino
            if self.fusao == "atencao":
                x_glm2, pesos = self.cruzada(x_glm2, arco, chaves, ignorar, prior, cls_tok, rascunho_cel)
            else:
                x_glm2 = self.film(x_glm2, glob, chaves, ignorar)
        if self.residual:
            d = self.delta(torch.cat([x_glm2_3d, x_glm2 - x_glm2_3d], dim=1)).transpose(2, 1)
            return torch.softmax(z3 + d, dim=-1), pesos
        return self._cabeca(x_glm2, x_ftm, x_mlp2, n_pts, self.cab_fus if self.assimetrica else None), pesos

    def _cabeca(self, x_glm2, x_ftm, x_mlp2, n_pts, h=None, logits=False):
        """Cabeça do MeshSegNet; h = ModuleDict com as mesmas camadas (cab_fus) ou None (as do próprio modelo)."""
        h = h if h is not None else {k: getattr(self, k) for k in CABECA}
        x = torch.max(x_glm2, 2, keepdim=True)[0]
        x = torch.nn.Upsample(n_pts)(x)
        x = torch.cat([x, x_ftm, x_mlp2, x_glm2], dim=1)
        x = F.relu(h["mlp3_bn1_1"](h["mlp3_conv1"](x)))
        x = F.relu(h["mlp3_bn1_2"](h["mlp3_conv2"](x)))
        x = F.relu(h["mlp3_bn2_1"](h["mlp3_conv3"](x)))
        if self.with_dropout:
            x = self.dropout(x)
        x = F.relu(h["mlp3_bn2_2"](h["mlp3_conv4"](x)))
        x = h["output_conv"](x).transpose(2, 1).contiguous()
        return x if logits else torch.softmax(x, dim=-1)


def carregar_pesos_iniciais(modelo, caminho):
    """Carrega um checkpoint pré-treinado (model_state_dict) nas camadas herdadas; as novas ficam como estão."""
    ck = torch.load(caminho, map_location="cpu", weights_only=True)
    faltando, sobrando = modelo.load_state_dict(ck["model_state_dict"], strict=False)
    novos = NOVOS
    faltando = [k for k in faltando if k.split(".")[0] not in novos]
    if faltando or sobrando:
        raise ValueError(f"checkpoint não casa com o MeshSegNet: faltando={faltando[:5]} sobrando={sobrando[:5]}")
    return modelo
