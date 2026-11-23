"""Dataset da fusão 2D → 3D: uma arcada decimada + os tokens 2D por dente da mesma arcada.

3D (preparar_fusao.py): as 15 features por célula do MeshSegNet, a coordenada do arco e o rótulo do
planejador (17 classes own_class). Augmentation só no treino: rotação 3D uniforme + escala 0,9–1,1,
O "arco" é calculado antes da rotação e não muda com ela.
Espelho L↔R, com probabilidade `espelho` no treino: os DOIS exames juntos. O 3D vira
x → −x (ordem dos vértices invertida, normal com x trocado), rótulo l → 17 − l, arco → 1 − arco; o 2D vem
da panorâmica invertida passada de novo no Mask R-CNN (extrair_features_2d.py --espelho), porque inverter
só pos_norm e probs deixaria o feat_box incoerente. Sem o .npz espelhado do caso, a amostra não espelha.

2D (extrair_features_2d.py, Mask R-CNN v2 congelado, pipeline NOVO): os dentes da panorâmica com
arcada igual à da malha (0 = superior, 1 = inferior). Cada token leva feat_box (1024), feat_mask (256),
probs (32), score, área relativa e pos_norm (x, y). É a saída CRUA do v2, nunca o planejador nem o
CVAT: no teste a fusão recebe o que o 2D realmente produz.
Caso sem .npz 2D (panorâmica pulada) → nenhum token: o modelo cai no token nulo, como no dropout de modalidade.

Uso: from dados_fusao import DatasetFusao, juntar
"""
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

sys.path[:0] = [str(p) for p in sorted(Path(__file__).resolve().parents[1].glob("0*_*"))]  # pastas do repositório
import caminhos  # noqa: E402
import malha  # noqa: E402

FEATURES_2D = caminhos.FEATURES_2D
FEATURES_2D_ESPELHO = caminhos.FEATURES_2D_ESPELHO
SIGMA_DENTE = 0.5     # σ do prior em larguras de dente: meia largura (as células da coroa ficam a ±0,5 do centro)


def dobra_do_caso(case_id, n_dobras):
    """Dobra estável de um caso (md5 do case_id), independente de quais manifestos entram.
    As duas arcadas do caso caem na mesma dobra."""
    return int(hashlib.md5(str(int(case_id)).encode()).hexdigest(), 16) % n_dobras


def _dobra(txt):
    k, n = (int(v) for v in txt.split("/"))
    assert 0 <= k < n, txt
    return k, n


def prior_registro(pasta, case_id, arch, arco, n_tok):
    """Prior da atenção guiada a partir do registro_fusao.py (sem rótulo; vale no treino e no teste):
    centro f̂(s) por célula, x da coroa por token (alinhado com tokens_2d), σ e se há prior.
    σ = largura · √(0,5² + rms²), com rms o resíduo dos pares casados em larguras de dente: um registro
    que encaixa mal deixa o prior mais largo (mais fraco). Sem registro → tem = False (atenção livre)."""
    p = Path(pasta) / f"{case_id}_{arch}.npz"
    if not p.exists():
        return np.zeros_like(arco), np.zeros(n_tok, np.float32), 1.0, False
    r = np.load(p)
    w = float(r["largura"])
    s = dict(zip(r["classe_mancha"].tolist(), r["s_mancha"].tolist()))
    res = [(np.polyval(r["coef"], s[a]) - r["x_tok"][j]) / w for a, j in r["pares"]]
    sigma = w * float(np.sqrt(SIGMA_DENTE ** 2 + (np.mean(np.square(res)) if res else 0.0)))
    return (np.polyval(r["coef"], arco).astype(np.float32), r["x_tok"][:n_tok].astype(np.float32), sigma, True)
T_MAX = 20            # dentes por arcada na panorâmica (16 + folga para extras/decíduos)
ORDER16 = {           # classe 1..16 → FDI (own_class, igual ao treinar_fusao.py)
    "upper": [18, 17, 16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26, 27, 28],
    "lower": [48, 47, 46, 45, 44, 43, 42, 41, 31, 32, 33, 34, 35, 36, 37, 38],
}
D_TOKEN_IN = 1024 + 256 + 32 + 4      # feat_box + feat_mask + probs + (score, área, x, y)
ARCADA_2D = {"upper": 0, "lower": 1}


def _rotacao():
    """Rotação 3D aleatória uniforme a partir de um quaternion."""
    q = np.random.normal(size=4)
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _rotacao_limitada(max_vertical, max_incl):
    """As malhas já vêm na convenção do modelo (Y apical, +Z anterior, conferido em 05/11) e a
    inferência também (orientação pelo par, erro residual 3–8°), então basta robustez a desvios pequenos: giro de
    ±max_vertical graus em torno de Y e inclinação de ±max_incl graus em torno de X e de Z."""
    def eixo(k, graus):
        a = np.radians(np.random.uniform(-graus, graus))
        c, s = np.cos(a), np.sin(a)
        i, j = [(1, 2), (2, 0), (0, 1)][k]
        R = np.eye(3)
        R[i, i], R[i, j], R[j, i], R[j, j] = c, -s, s, c
        return R
    return eixo(1, max_vertical) @ eixo(0, max_incl) @ eixo(2, max_incl)


def features_15(verts, faces, bary, normals):
    """As 15 features por célula do MeshSegNet (00_comum/malha.py)."""
    return malha.extract_features(verts, faces, bary, normals)


def espelhar(verts, bary, faces, label, arco, normals):
    """Espelho sagital do 3D (x → −x): a malha continua orientada (ordem dos vértices invertida), o dente
    troca de lado (own_class l → 17 − l: 18 → 28, 46 → 36) e o arco recomeça do novo lado direito."""
    m = np.array([-1.0, 1.0, 1.0])
    return (verts * m, bary * m, faces[:, ::-1].copy(), np.where(label > 0, 17 - label, 0), 1 - arco,
            (normals * m).astype(normals.dtype))


def tokens_2d(case_id, arch, pasta=FEATURES_2D):
    """(tokens T×D_TOKEN_IN, fdi T, fpn_global 1280) da panorâmica do caso; T = 0 se não houver."""
    p = Path(pasta) / f"{case_id}.npz"
    if not p.exists():
        return np.zeros((0, D_TOKEN_IN), np.float32), np.zeros(0, np.int16), np.zeros(1280, np.float32)
    z = np.load(p)
    sel = z["arcada"] == ARCADA_2D[arch]
    area = z["area"][sel].astype(np.float32) / float(np.prod(z["tamanho"]))
    tok = np.concatenate([z["feat_box"][sel].astype(np.float32), z["feat_mask"][sel].astype(np.float32),
                          z["probs"][sel].astype(np.float32),
                          np.stack([z["score"][sel], area], 1).astype(np.float32),
                          z["pos_norm"][sel].astype(np.float32)], axis=1)
    ordem = np.argsort(z["pos_norm"][sel][:, 0])          # da esquerda para a direita na imagem
    return tok[ordem][:T_MAX], z["fdi"][sel][ordem][:T_MAX], z["fpn_global"].astype(np.float32)


class DatasetFusao(Dataset):
    """Uma amostra = uma arcada. patch=None devolve a malha inteira (avaliação, batch 1)."""

    def __init__(self, raiz, split, arch, patch=7000, augment=False, fundo=0.25, pasta_2d=FEATURES_2D,
                 espelho=0.0, pasta_2d_espelho=FEATURES_2D_ESPELHO, pasta_registro=None, dificil=0.0,
                 com_complementar=False, rot_max=None, rot_incl=5.0, com_ampliacao=False,
                 sem_dobra=None, so_dobra=None):
        raiz = Path(raiz)
        # rot_max=None mantém a rotação 3D uniforme de antes (Z1/Z2 e runs antigos); um número limita
        # o giro em torno da vertical (±rot_max) e a inclinação (±rot_incl).
        self.rot_max, self.rot_incl = rot_max, rot_incl
        man = pd.read_csv(raiz / "manifesto.csv")
        # Os 121 casos do treino do MeshSegNet pré-treinado (manifesto_complementar.csv, todos com split = treino),
        # só quando pedido: assim os runs antigos e os novos comparam o efeito de ter esses casos.
        if com_complementar and (raiz / "manifesto_complementar.csv").exists():
            man = pd.concat([man, pd.read_csv(raiz / "manifesto_complementar.csv")], ignore_index=True)
        # A ampliação (preparar_fusao.py --lote ampliacao), toda no treino.
        if com_ampliacao and (raiz / "manifesto_ampliacao.csv").exists():
            novos = pd.read_csv(raiz / "manifesto_ampliacao.csv")
            fora = set(man.case_id[man.split != "treino"])
            assert not fora & set(novos.case_id), "caso da ampliação no val/teste"
            man = pd.concat([man, novos], ignore_index=True)
        self.linhas = man[(man.split == split) & (man.arch == arch)].reset_index(drop=True)
        # Validação cruzada para os rascunhos 3D. sem_dobra="k/K" tira a dobra k (treino da dobra);
        # so_dobra="k/K" fica só com ela (predição fora da dobra). Só faz sentido no split de treino.
        for txt, manter in ((sem_dobra, False), (so_dobra, True)):
            if txt:
                assert split == "treino", "dobras só no split de treino"
                k, n = _dobra(txt)
                na_k = self.linhas.case_id.map(lambda c: dobra_do_caso(c, n) == k)
                self.linhas = self.linhas[na_k == manter].reset_index(drop=True)
        self.raiz, self.arch, self.patch, self.augment, self.fundo = raiz, arch, patch, augment, fundo
        # Fração do patch tirada das células em que o rascunho 3D erra (rascunho_label ≠ label, ~7% da
        # malha). Só no treino: o GT entra como rótulo, como sempre; na inferência nada disso existe.
        self.dificil = dificil
        self.pasta_2d, self.espelho, self.pasta_2d_espelho = pasta_2d, espelho, Path(pasta_2d_espelho)
        self.pasta_registro = Path(pasta_registro) if pasta_registro else raiz / "registro"

    def __len__(self):
        return len(self.linhas)

    def _amostrar(self, label, erro=None):
        """Patch balanceado: `fundo` de gengiva e o resto dividido igualmente entre os dentes.
        Com `dificil` > 0, essa fração do patch vem antes das células `erro` (onde o rascunho 3D erra)."""
        n = len(label)
        if self.patch is None:
            return np.arange(n)
        gengiva, dentes = np.flatnonzero(label == 0), [c for c in np.unique(label) if c > 0]
        if not len(gengiva) or not dentes:
            return np.random.choice(n, self.patch, replace=n < self.patch)
        sel, patch = [], self.patch
        if self.dificil > 0 and erro is not None and erro.any():
            cand_e = np.flatnonzero(erro)
            n_e = int(round(self.patch * self.dificil))
            sel.append(np.random.choice(cand_e, n_e, replace=len(cand_e) < n_e))
            patch -= n_e
        n_g = int(round(patch * self.fundo))
        base, resto = divmod(patch - n_g, len(dentes))
        sel.append(np.random.choice(gengiva, n_g, replace=len(gengiva) < n_g))
        for k, c in enumerate(dentes):
            cand, q = np.flatnonzero(label == c), base + int(k < resto)
            sel.append(np.random.choice(cand, q, replace=len(cand) < q))
        sel = np.concatenate(sel)
        np.random.shuffle(sel)
        return sel

    def __getitem__(self, i):
        r = self.linhas.iloc[i]
        z = np.load(self.raiz / "malhas" / r.arquivo)
        verts, bary, normals = z["verts"].astype(np.float64), z["bary"].astype(np.float64), z["normals"]
        faces, label, arco = z["faces"].astype(np.int64), z["label"].astype(np.int64), z["arco"].astype(np.float32)
        # O "plano B" (sem rascunho_label → cópia do GT) virou erro; nunca foi acionado, mas seria vazamento.
        if "rascunho_label" not in z.files:
            raise KeyError(f"{r.arquivo} sem rascunho_label (rascunho 3D); o GT nunca substitui o rascunho")
        zl = z["rascunho_label"].astype(np.int64)
        pasta_2d, espelhado = self.pasta_2d, False
        if (self.augment and self.espelho and np.random.rand() < self.espelho
                and (self.pasta_2d_espelho / f"{int(r.case_id)}.npz").exists()):
            verts, bary, faces, label, arco, normals = espelhar(verts, bary, faces, label, arco, normals)
            zl = np.where(zl > 0, 17 - zl, 0)
            pasta_2d, espelhado = self.pasta_2d_espelho, True
        erro = zl != label
        if self.augment:
            R = _rotacao() if self.rot_max is None else _rotacao_limitada(self.rot_max, self.rot_incl)
            c = verts.mean(axis=0)
            s = np.random.uniform(0.9, 1.1)
            verts, bary = (verts - c) @ R.T * s + c, (bary - c) @ R.T * s + c
            normals = (normals.astype(np.float64) @ R.T).astype(np.float32)
        feat = features_15(verts, faces, bary, normals)
        sel = self._amostrar(label, erro)
        tok, fdi2d, glob = tokens_2d(int(r.case_id), self.arch, pasta_2d)
        # O registro é do caso original; no espelhado os tokens são outros, então a amostra vai sem prior.
        centro, x_tok, sigma, tem = prior_registro(self.pasta_registro, int(r.case_id), self.arch, arco, len(tok))
        tem = tem and not espelhado and len(x_tok) == len(tok)
        # Classe 0..16 de cada token (−1 se o FDI não é desta arcada) e a predição crua do rascunho 3D por
        # célula: o portão por célula compara "para que dente a célula olha" com "o que o rascunho 3D disse". Sem rótulo.
        f2c = {f: k + 1 for k, f in enumerate(ORDER16[self.arch])}
        cls_tok = np.array([f2c.get(int(f), -1) for f in fdi2d], np.int64)
        return dict(cells=torch.from_numpy(feat[sel].T.copy()), labels=torch.from_numpy(label[sel]),
                    erro_rascunho=torch.from_numpy(erro[sel]), rascunho_label=torch.from_numpy(zl[sel]),
                    cls_tok=torch.from_numpy(cls_tok),
                    arco=torch.from_numpy(arco[sel]), centro=torch.from_numpy(centro[sel]),
                    x_tok=torch.from_numpy(x_tok), sigma=torch.tensor(sigma, dtype=torch.float32),
                    tem_prior=torch.tensor(tem),
                    tokens=torch.from_numpy(tok), fdi_2d=torch.from_numpy(fdi2d.astype(np.int64)),
                    glob=torch.from_numpy(glob), case_id=int(r.case_id))


def juntar(amostras):
    """Collate: células já têm o mesmo patch; os tokens são completados até T_MAX com máscara."""
    B = len(amostras)
    tokens = torch.zeros(B, T_MAX, D_TOKEN_IN)
    fdi = torch.zeros(B, T_MAX, dtype=torch.long)
    valido = torch.zeros(B, T_MAX, dtype=torch.bool)
    x_tok = torch.zeros(B, T_MAX)
    cls_tok = torch.full((B, T_MAX), -1, dtype=torch.long)
    for b, s in enumerate(amostras):
        t = len(s["tokens"])
        tokens[b, :t], fdi[b, :t], valido[b, :t] = s["tokens"], s["fdi_2d"], True
        cls_tok[b, :t] = s["cls_tok"]
        x_tok[b, :len(s["x_tok"])] = s["x_tok"]
    return dict(cells=torch.stack([s["cells"] for s in amostras]), labels=torch.stack([s["labels"] for s in amostras]),
                erro_rascunho=torch.stack([s["erro_rascunho"] for s in amostras]),
                rascunho_label=torch.stack([s["rascunho_label"] for s in amostras]), cls_tok=cls_tok,
                arco=torch.stack([s["arco"] for s in amostras]), tokens=tokens, fdi_2d=fdi, valido=valido,
                centro=torch.stack([s["centro"] for s in amostras]), x_tok=x_tok,
                sigma=torch.stack([s["sigma"] for s in amostras]), tem_prior=torch.stack([s["tem_prior"] for s in amostras]),
                glob=torch.stack([s["glob"] for s in amostras]), case_id=[s["case_id"] for s in amostras])
