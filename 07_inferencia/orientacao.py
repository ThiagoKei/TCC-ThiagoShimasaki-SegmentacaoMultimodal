"""Orientação do escaneamento no referencial em que o modelo foi treinado (a arcada "em pé").

Referencial do treino: Y apical-oclusal (na arcada INFERIOR as coroas apontam para +Y e a base para −Y),
+Z anterior (incisivos), X = Y × Z (transversal). Os modelos só viram desvios de ±10° em torno de Y e ±5° de
inclinação, então a orientação precisa chegar perto disso.

Dois caminhos:
  * com a SUPERIOR também: orientação pelo par (ferramenta externa, não incluída);
  * só a INFERIOR: orienta_inferior (abaixo):
      vertical = eixo de menor variância (área-ponderada) com o sinal dado pela BORDA da malha (o escaneamento
      é aberto na base, então a borda fica embaixo; 97 % de acerto nas 481 arcadas do TCC) e, na falta de borda,
      pela soma das normais; o sinal final é decidido no rodar.py pelo G4-S (a arcada de ponta-cabeça dá
      confiança menor: 139/139 no teste);
      anteroposterior = eixo do meio no plano oclusal, com o sinal pela forma do U (a ponta fechada é a frente);
      transversal = Y × Z; e uma correção fixa CALIBRADA nas malhas de treino (o plano de menor variância de uma
      arcada não é exatamente o plano oclusal do planejador).
"""
import numpy as np

AMOSTRA = 20000
FATIA_PONTA = 0.12
RAZAO_PONTA_MAX = 0.6
# Correção média (eixos estimados → referencial do treino), calibrada em calibrar_orientacao.
CORRECAO = np.array([[0.99991, -0.00263, 0.01296],
                     [0.00348, 0.99783, -0.06581],
                     [-0.01276, 0.06584, 0.99775]])   # 3,8°, 218 arcadas de treino


class OrientacaoIncerta(Exception):
    pass


def _amostra_area(verts, faces, n, seed=0):
    tri = verts[faces]
    cr = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = 0.5 * np.linalg.norm(cr, axis=1)
    bary = tri.mean(1)
    p = area / area.sum()
    idx = np.random.default_rng(seed).choice(len(faces), min(n, len(faces)), replace=len(faces) < n, p=p)
    return bary[idx], cr.sum(0), bary, area


def borda(faces):
    """Vértices das arestas de borda (arestas que pertencem a uma face só)."""
    e = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    u, cnt = np.unique(e, axis=0, return_counts=True)
    return np.unique(u[cnt == 1])


def eixos_inferior(verts, faces):
    """(R sem correção, centro, razão das pontas). R: linhas = X, Y, Z no referencial de entrada."""
    verts = np.asarray(verts, np.float64)
    faces = np.asarray(faces, np.int64)
    pts, soma_normais, _, _ = _amostra_area(verts, faces, AMOSTRA)
    c = pts.mean(0)
    q = pts - c
    w, vecs = np.linalg.eigh(np.cov(q.T))
    y = vecs[:, 0]
    bv = borda(faces)                             # o escaneamento é aberto na base: a borda fica embaixo
    if len(bv) >= 20:
        if (verts[bv].mean(0) - c) @ y > 0:
            y = -y
    elif soma_normais @ y < 0:                    # malha fechada: a soma das normais aponta para as coroas
        y = -y
    qp = q - np.outer(q @ y, y)
    w2, v2 = np.linalg.eigh(np.cov(qp.T))
    no_plano = sorted((v2[:, i] for i in range(3)), key=lambda e: abs(e @ y))[:2]
    a, b = no_plano
    xt, ap = (a, b) if np.var(qp @ a) >= np.var(qp @ b) else (b, a)
    s, t = qp @ ap, qp @ xt
    lo_q, hi_q = np.quantile(s, [FATIA_PONTA, 1.0 - FATIA_PONTA])
    meio = np.median(t)
    larg_lo = float(np.mean(np.abs(t[s <= lo_q] - meio)))
    larg_hi = float(np.mean(np.abs(t[s >= hi_q] - meio)))
    if max(larg_lo, larg_hi) < 1e-6:
        raise OrientacaoIncerta("arcada sem largura")
    razao = min(larg_lo, larg_hi) / max(larg_lo, larg_hi)
    z = ap if larg_hi < larg_lo else -ap          # a frente é a ponta estreita do U
    z = z - (z @ y) * y
    z /= np.linalg.norm(z)
    x = np.cross(y, z)
    return np.vstack([x, y, z]), c, razao


def orienta_inferior(verts, faces):
    """{"R": 3×3, "centro": 3, "razao_ponta": float}; p_orientado = R @ (p − centro)."""
    R, c, razao = eixos_inferior(verts, faces)
    # razão > RAZAO_PONTA_MAX: frente/trás pouco decididos pela forma; o rodar.py então deixa o G4-S decidir também
    # o sinal anteroposterior (candidatos extras).
    return {"R": CORRECAO @ R, "centro": c, "razao_ponta": razao, "ap_incerto": razao > RAZAO_PONTA_MAX}


def orienta(verts_inf, faces_inf, verts_sup=None):
    """Orienta pela arcada inferior (orienta_inferior). A orientação pelo par superior + inferior não faz parte
    deste repositório."""
    if verts_sup is not None:
        raise NotImplementedError("a orientação pelo par superior + inferior não faz parte deste repositório; "
                                  "rode só com a arcada inferior")
    r = orienta_inferior(verts_inf, faces_inf)
    r["metodo"] = "só a inferior (orienta_inferior)"
    return r


def angulo_graus(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))
