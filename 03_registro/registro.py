"""Registro panorâmica ↔ malha sem rótulo: casa a ordem dos dentes do rascunho 3D com a dos dentes detectados
na panorâmica. Usado pelo treino (03_registro/registro_fusao.py, em todas as arcadas) e pela inferência
(07_inferencia/rodar.py, numa arcada).

Cada "mancha" 3D (uma classe do rascunho com >= 1% das células de dente) tem uma posição s no arco (0 = lado direito
do paciente); cada dente da panorâmica tem a posição x da coroa. `registrar` acha o alinhamento que preserva a ordem
(ICP 1D, pulando dos dois lados) e a curva x = f(s) (polinômio de grau <= 3) que vira o prior da atenção guiada.
"""
import itertools

import numpy as np
from pycocotools import mask as mu

QUAD = {"upper": (1, 2), "lower": (3, 4)}
PULO = 0.6
BETA = 1.0
MIN_FACES = 0.01
COROA = 0.4
T_MAX = 20
ARCADA_2D = {"upper": 0, "lower": 1}
ORDER16 = {"upper": [18, 17, 16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26, 27, 28],
           "lower": [48, 47, 46, 45, 44, 43, 42, 41, 31, 32, 33, 34, 35, 36, 37, 38]}


def coroa_2d(f, m):
    ys, xs = np.nonzero(m)
    h = ys.max() - ys.min() + 1
    sup = f // 10 in (1, 2)
    occ = ys >= ys.max() - COROA * h if sup else ys <= ys.min() + COROA * h
    return xs[occ].mean(), np.quantile(ys, 0.98 if sup else 0.02), xs.max() - xs.min() + 1


def alinhar(fs, x, w, discorda=None, beta=BETA, pulo=PULO):
    n, m = len(fs), len(x)
    C = np.abs(fs[:, None] - x[None, :]) / w
    if discorda is not None:
        C = C + beta * discorda
    D = np.zeros((n + 1, m + 1))
    D[:, 0] = np.arange(n + 1) * pulo
    D[0, :] = np.arange(m + 1) * pulo
    B = np.zeros((n + 1, m + 1), np.int8)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            op = (D[i - 1, j - 1] + C[i - 1, j - 1], D[i - 1, j] + pulo, D[i, j - 1] + pulo)
            B[i, j] = int(np.argmin(op))
            D[i, j] = op[B[i, j]]
    pares, i, j = [], n, m
    while i and j:
        if B[i, j] == 0:
            pares.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif B[i, j] == 1:
            i -= 1
        else:
            j -= 1
    return D[n, m], pares[::-1]


def ajuste_livre(s, x):
    return np.polyfit(s, x, 3 if len(s) >= 8 else 1)


def registrar(s, x, w, discorda=None, beta=BETA, iters=8, ajuste=ajuste_livre, inits=()):
    melhor = (np.inf, None, None)
    iniciais = []
    for a3, b3, a2, b2 in itertools.product((0, 1), (0, 1), (0, 1), (0, 1)):
        s0, s1 = s[a3], s[len(s) - 1 - b3]
        x0, x1 = x[a2], x[len(x) - 1 - b2]
        if s1 <= s0 or x1 <= x0:
            continue
        iniciais.append(np.array([(x1 - x0) / (s1 - s0), x0 - s0 * (x1 - x0) / (s1 - s0)]))
    for coef in iniciais + list(inits):
        for _ in range(iters):
            custo, pares = alinhar(np.polyval(coef, s), x, w, discorda, beta)
            if len(pares) < 3:
                break
            i, j = np.array(pares).T
            novo = ajuste(s[i], x[j])
            if np.allclose(novo[-2:], coef[-2:]) and len(novo) == len(coef):
                coef = novo
                break
            coef = novo
        custo, pares = alinhar(np.polyval(coef, s), x, w, discorda, beta)
        if custo < melhor[0]:
            melhor = (custo, coef, pares)
    return melhor


def manchas(arco, rascunho_label):
    dentes = rascunho_label > 0
    cls = [c for c in np.unique(rascunho_label[dentes]) if (rascunho_label == c).sum() >= MIN_FACES * dentes.sum()]
    return np.array(cls, int), np.array([np.median(arco[rascunho_label == c]) for c in cls])


def deteccoes(f2d, arch, com_mascaras=False):
    """Detecções da arcada na ordem dos tokens: fdi, x da coroa e largura normalizados (e as máscaras 512², se
    pedidas). Uma entrada por token, alinhada com o dados_fusao.tokens_2d; máscara vazia → x e largura NaN."""
    sel = np.flatnonzero(f2d["arcada"] == ARCADA_2D[arch])
    sel = sel[np.argsort(f2d["pos_norm"][sel, 0])][:T_MAX]
    _, x0, _, cw, _ = f2d["recorte"]
    H, W = (int(v) for v in f2d["tamanho"])
    fdi, mascaras, x, larg = [], [], [], []
    for k in sel:
        m = mu.decode({"size": [H, W], "counts": str(f2d["mask_rle"][k])}).astype(bool)
        cx, _, wd = coroa_2d(int(f2d["fdi"][k]), m) if m.any() else (np.nan, None, np.nan)
        fdi.append(int(f2d["fdi"][k])), mascaras.append(m), x.append((cx - x0) / cw), larg.append(wd / cw)
    if com_mascaras:
        return np.array(fdi, int), mascaras, np.array(x), np.array(larg)
    return np.array(fdi, int), np.array(x), np.array(larg)


def orientar_arco(arco, rascunho):
    """registro_fusao --rotulos: classes 1..8 (lado direito do paciente) ficam no começo do arco."""
    d_n, e_n = (rascunho >= 1) & (rascunho <= 8), rascunho >= 9
    inv = bool(d_n.any() and e_n.any() and arco[d_n].mean() > arco[e_n].mean())
    return ((1.0 - arco).astype(np.float32) if inv else arco), inv


def registrar_arcada(arco, rascunho, f2d, arch="lower"):
    """dict com coef, x_tok, largura, pares, s_mancha, classe_mancha (o .npz do registro) ou (None, motivo)."""
    cls, s = manchas(arco, rascunho)
    fdi_tok, x_tok, larg_tok = deteccoes(f2d, arch)
    v = np.flatnonzero(~np.isnan(x_tok))
    fdi2, x, larg = fdi_tok[v], x_tok[v], larg_tok[v]
    if len(cls) < 4 or len(fdi2) < 4:
        return None, f"menos de 4 dentes no 3D ({len(cls)}) ou na panorâmica ({len(fdi2)})"
    w = float(np.median(larg))
    o3, o2 = np.argsort(s), np.argsort(x)
    fdi3 = np.array([ORDER16[arch][c - 1] for c in cls])
    discorda = (fdi3[o3][:, None] != fdi2[o2][None, :]).astype(float)
    custo, coef, pares = registrar(s[o3], x[o2], w, discorda)
    if coef is None:
        return None, "registro sem pares"
    coef = np.r_[np.zeros(4 - len(coef)), coef]
    pares = np.array([(cls[o3[i]], o2[j]) for i, j in pares], int)
    return dict(coef=coef, x_tok=x_tok, largura=w, pares=np.c_[pares[:, 0], v[pares[:, 1]]], s_mancha=s,
                classe_mancha=cls), f"{len(pares)} pares casados"
