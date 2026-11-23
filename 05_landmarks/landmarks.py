"""Landmarks por dente na inferência (modelo "so3d", média das 3 sementes).

O recorte de cada dente usa o referencial do recortes_dentes.py (o mesmo do treino): 512 pontos com 2 mm de gengiva
em volta. `recortes` também é usado pelo teste_3dteethland.py.

Para cada dente segmentado: recorte no referencial do dente (x distal, y apical, z vestibular) → rede →
eixo longo IE (vetor unitário), pontos Mesial, Distal, FacialPoint, InnerPoint, OuterPoint e cúspides.
"""
import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree

from modelo_landmarks import CANAL_DENTE, ModeloLandmarks, extrair_pontos
from recortes_dentes import GRUPO, MARGEM, N_PONTOS, referencial
from treinar_landmarks import ESC, UNICOS, condicao

ARCH = "lower"
ORDER16 = [48, 47, 46, 45, 44, 43, 42, 41, 31, 32, 33, 34, 35, 36, 37, 38]


def recortes(bary, nrm, lab, arch, rng):
    """{classe: recorte} de cada dente da arcada: origem o, base R (x distal, y apical, z vestibular), células
    usadas e os 512 pontos (xyz e normal no referencial do dente + "é dente")."""
    arv, idx_g = cKDTree(bary[lab == 0]), np.flatnonzero(lab == 0)
    centro = bary[lab > 0].mean(0)
    saida = {}
    for classe in sorted(int(c) for c in np.unique(lab) if c > 0):
        o, R = referencial(bary, lab, classe, arch, centro)
        if o is None:
            continue
        dente = np.flatnonzero(lab == classe)
        perto = arv.query_ball_point(bary[dente], MARGEM)
        geng = idx_g[np.unique(np.concatenate([np.asarray(p, int) for p in perto]))] if len(perto) else []
        cel = np.r_[dente, geng].astype(int)
        c = cel[rng.choice(len(cel), N_PONTOS, replace=len(cel) < N_PONTOS)]
        pts = np.c_[(bary[c] - o) @ R.T, nrm[c] @ R.T, np.isin(c, dente).astype(float)].astype(np.float32)
        saida[classe] = dict(o=o, R=R, cel=c, pts=pts, n_cel=len(dente))
    return saida


def carregar_modelos(caminhos):
    ms = []
    for p in caminhos:
        c = torch.load(p, map_location="cpu", weights_only=False)
        m = ModeloLandmarks(com_2d=False)
        m.load_state_dict(c["modelo"])
        ms.append(m.eval())
    return ms


@torch.no_grad()
def landmarks(modelos, rec):
    classes = list(rec)
    if not classes:
        return {}
    tipo = np.array([8 - min(c - 1, 16 - c) for c in classes])
    cond = torch.from_numpy(condicao(pd.DataFrame(dict(tipo=tipo, grupo=[GRUPO[t] for t in tipo],
                                                       arch=[ARCH] * len(classes)))))
    pts = torch.from_numpy(np.stack([rec[c]["pts"] for c in classes]))
    ent = pts.clone()
    ent[..., :3] /= ESC
    soma_ie, soma_lg, soma_ds = 0, 0, 0
    for m in modelos:
        lg, ds, ie = m(ent, cond)
        soma_ie, soma_lg, soma_ds = soma_ie + ie, soma_lg + lg, soma_ds + ds
    ie = torch.nn.functional.normalize(soma_ie, dim=1).numpy()
    lg, ds = (soma_lg / len(modelos)).numpy(), (soma_ds / len(modelos)).numpy() * ESC
    out = {}
    for k, c in enumerate(classes):
        xyz = rec[c]["pts"][:, :3]
        uni, sc, cus = extrair_pontos(xyz, lg[k][:, :CANAL_DENTE], ds[k])
        out[c] = dict(ie=ie[k], uni=uni, sc=sc, cus=cus, dente_ref=lg[k][:, CANAL_DENTE] > 0, tipo=int(tipo[k]))
    return out


def landmarks_globais(modelos, bary, nrm, lab):
    """{fdi: {...}} no referencial em que bary/nrm estão (o orientado)."""
    rec = recortes(bary, nrm, lab, ARCH, np.random.default_rng(0))
    lm = landmarks(modelos, rec)
    saida = {}
    for classe, L in lm.items():
        R, o = rec[classe]["R"], rec[classe]["o"]
        glob = lambda p: o + R.T @ np.asarray(p)          # noqa: E731
        saida[ORDER16[classe - 1]] = dict(
            eixo_IE=R.T @ L["ie"], centro=o,
            pontos={k: glob(L["uni"][j]) for j, k in enumerate(UNICOS)},
            cuspides=[glob(p) for p, _ in L["cus"]], n_celulas=rec[classe]["n_cel"])
    return saida
