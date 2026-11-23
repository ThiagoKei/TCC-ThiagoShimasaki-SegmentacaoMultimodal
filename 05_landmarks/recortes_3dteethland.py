"""Recortes por dente do 3DTeethLand (pontos da coroa) no MESMO formato dos recortes_dentes.py.

Para o treino conjunto do modelo de landmarks: o mesmo codificador por dente, uma cabeça para os pontos da coroa
(treinada aqui, sem 2D) e outra para o eixo longo IE (casos do planejador, com e sem panorâmica).

Entrada:
  malha     <TCC_DADOS>/teeth3ds/prep/<ID>_<arch>.npz (01_dados/teeth3ds.py preparar --conjunto landmarks_treino: girada para a
            convenção do modelo, decimada para ~8.000 células, rótulo 0..16 por célula)
  rótulo    <TCC_DADOS>/teeth3ds/scans/<arch>/<ID>/<ID>_<arch>.npz (FDI por vértice da malha cheia)
  pontos    3DTeethLand_landmarks_{train,test}.zip → <arch>/<ID>/<ID>_<arch>__kpt.json (classe + coordenada; o JSON NÃO
            diz de que dente é cada ponto)

Atribuição ponto → dente (na malha cheia, girada com a mesma matriz do preparo):
  Mesial / Distal  ficam no contato entre vizinhos: entre os dentes com vértice a até RAIO mm, fica o dente em que o ponto
                   está mais para o lado certo (x local negativo = mesial, positivo = distal), normalizado pela largura
  outros           o dente do vértice de dente mais próximo (até RAIO mm)
Referencial e recorte: os MESMOS do recortes_dentes.py (referencial(), N_PONTOS, MARGEM), pela segmentação GT.

Saída: <TCC_DADOS>/fusao_dados/landmarks/recortes_3dtl.npz (pontos (N, 512, 7); kp (N, 11, 3) no referencial
local, em mm; kp_mask (N, 11)) e recortes_3dtl.csv (uma linha por dente). Slots de kp: KP_CLASSES (cúspides até 6,
ordenadas da mais alta para a mais baixa).
Uso: python 05_landmarks/recortes_3dteethland.py [--n N]
"""
import argparse
import json
import sys
import zipfile
from pathlib import Path

AQUI = Path(__file__).resolve().parent
sys.path[:0] = [str(p) for p in sorted(Path(__file__).resolve().parents[1].glob("0*_*"))]  # pastas do repositório
import caminhos  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

from dados_fusao import ORDER16  # noqa: E402
from recortes_dentes import GRUPO, MARGEM, N_PONTOS, SAIDA, referencial  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
T3D = caminhos.TEETH3DS
PREP = T3D / "prep"
# Mesma rotação do 01_dados/teeth3ds.py (x' = sx·x, y' = sy·z, z' = sz·y); repetida aqui para não importar o teeth3ds.py.
ROTACAO = {"upper": (-1, -1, -1), "lower": (1, 1, -1)}
UNICOS = ["Mesial", "Distal", "FacialPoint", "InnerPoint", "OuterPoint"]
MAX_CUSP = 6
KP_CLASSES = UNICOS + [f"Cusp{i}" for i in range(1, MAX_CUSP + 1)]
RAIO = 1.5


def matriz(s):
    sx, sy, sz = s
    return np.array([[sx, 0, 0], [0, 0, sy], [0, sz, 0]], dtype=np.float64)


def ler_kpts(conjunto):
    """{(ID, arch): [(classe, xyz), ...]} dos dois zips."""
    zp = T3D / f"3DTeethLand_landmarks_{'train' if conjunto == 'treino' else 'test'}.zip"
    out = {}
    with zipfile.ZipFile(zp) as z:
        for n in z.namelist():
            if n.endswith("__kpt.json"):
                cid, arch = Path(n).name.replace("__kpt.json", "").rsplit("_", 1)
                out[(cid, arch)] = [(o["class"], np.array(o["coord"], float)) for o in json.loads(z.read(n))["objects"]]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int)
    a = ap.parse_args()
    kpts = ler_kpts("treino")
    alvo = sorted(k for k in kpts if (PREP / f"{k[0]}_{k[1]}.npz").exists())
    print(f"{len(kpts)} scans com pontos de treino; {len(alvo)} já preparados", flush=True)
    if a.n:
        alvo = alvo[: a.n]
    rng = np.random.default_rng(0)
    P, K, M, L, dist_sup, sem_dente, ambiguos = [], [], [], [], [], 0, 0
    for n, (cid, arch) in enumerate(alvo, 1):
        R0 = matriz(ROTACAO[arch])
        d = np.load(PREP / f"{cid}_{arch}.npz")
        bary, nrm, label = d["bary"].astype(float), d["normals"].astype(float), d["label"].astype(int)
        cheia = np.load(T3D / "scans" / arch / cid / f"{cid}_{arch}.npz")
        verts = cheia["verts"].astype(float) @ R0.T
        f2c = {f: i + 1 for i, f in enumerate(ORDER16[arch])}
        lab_v = np.array([f2c.get(int(f), 0) for f in cheia["labels"]])
        arv_v = cKDTree(verts)
        dente_v = lab_v > 0
        arv_d = cKDTree(verts[dente_v])
        cls_d = lab_v[dente_v]
        centro = bary[label > 0].mean(0)
        refs = {}
        for c in sorted(int(c) for c in np.unique(label) if c > 0):
            o, R = referencial(bary, label, c, arch, centro)
            if o is not None:
                larg = np.ptp((bary[label == c] - o) @ R[0])
                refs[c] = (o, R, max(larg, 1e-3))
        por_dente = {c: {k: [] for k in UNICOS + ["Cusp"]} for c in refs}
        for classe_kp, xyz in kpts[(cid, arch)]:
            p = R0 @ xyz
            dist_sup.append(arv_v.query(p)[0])
            vizinhos = arv_d.query_ball_point(p, RAIO)
            cand = sorted({int(cls_d[i]) for i in vizinhos} & set(refs))
            if not cand:
                sem_dente += 1
                continue
            if classe_kp in ("Mesial", "Distal"):
                sinal = -1.0 if classe_kp == "Mesial" else 1.0
                lado = {c: sinal * ((p - refs[c][0]) @ refs[c][1][0]) / refs[c][2] for c in cand}
                certos = [c for c in cand if lado[c] > 0]
                if len(certos) > 1:
                    # Linha média: o Mesial do 11 e o do 21 (31/41) ficam do lado mesial dos DOIS → o mais próximo
                    dist = {c: np.min(np.linalg.norm(verts[lab_v == c] - p, axis=1)) for c in certos}
                    c = min(dist, key=dist.get)
                else:
                    c = max(lado, key=lado.get)
                ambiguos += len(cand) > 1
            else:
                _, i = arv_d.query(p)
                c = int(cls_d[i]) if int(cls_d[i]) in refs else cand[0]
            o, R, _ = refs[c]
            chave = "Cusp" if classe_kp == "Cusp" else classe_kp
            if chave in por_dente[c]:
                por_dente[c][chave].append(R @ (p - o))
        arv_g = cKDTree(bary[label == 0])
        idx_g = np.flatnonzero(label == 0)
        for c, pts_kp in por_dente.items():
            if not any(len(v) for v in pts_kp.values()):
                continue                                           # dente sem nenhum ponto anotado
            o, R, _ = refs[c]
            dente = np.flatnonzero(label == c)
            perto = arv_g.query_ball_point(bary[dente], MARGEM)
            gengiva = idx_g[np.unique(np.concatenate([np.asarray(q, int) for q in perto]))] if len(perto) else []
            cel = np.r_[dente, gengiva].astype(int)
            sel = cel[rng.choice(len(cel), N_PONTOS, replace=len(cel) < N_PONTOS)]
            pts = np.c_[(bary[sel] - o) @ R.T, nrm[sel] @ R.T, np.isin(sel, dente).astype(float)]
            kp, mk = np.zeros((len(KP_CLASSES), 3)), np.zeros(len(KP_CLASSES), bool)
            n_rep = {}
            for j, k in enumerate(UNICOS):
                if pts_kp[k]:
                    kp[j], mk[j] = pts_kp[k][0], True
                n_rep[k] = len(pts_kp[k])
            cusp = sorted(pts_kp["Cusp"], key=lambda v: v[1])[:MAX_CUSP]   # y apical: a mais alta primeiro
            for j, v in enumerate(cusp):
                kp[len(UNICOS) + j], mk[len(UNICOS) + j] = v, True
            fdi = ORDER16[arch][c - 1]
            tipo = 8 - min(c - 1, 16 - c)
            P.append(pts.astype(np.float32))
            K.append(kp.astype(np.float32))
            M.append(mk)
            L.append(dict(id=cid, arch=arch, split="treino_3dtl", fdi=fdi, tipo=tipo, grupo=GRUPO[tipo],
                          n_cusp=len(pts_kp["Cusp"]), **{f"n_{k}": n_rep[k] for k in UNICOS}, n_celulas=len(dente)))
        if n % 25 == 0:
            print(f"[{n}/{len(alvo)}] {len(L)} dentes", flush=True)
    D = pd.DataFrame(L)
    SAIDA.mkdir(exist_ok=True)
    np.savez_compressed(SAIDA / "recortes_3dtl.npz", pontos=np.stack(P), kp=np.stack(K), kp_mask=np.stack(M),
                        kp_classes=np.array(KP_CLASSES))
    D.to_csv(SAIDA / "recortes_3dtl.csv", index=False)
    ds = np.array(dist_sup)
    print(f"\n{len(D)} dentes de {(D.id + D.arch).nunique()} scans ({D.id.nunique()} pacientes) | ponto→superfície: mediana {np.median(ds):.2f} mm, "
          f"p99 {np.percentile(ds, 99):.2f} mm | pontos sem dente a {RAIO} mm: {sem_dente} | M/D com >1 candidato: {ambiguos}")
    print("pontos por dente (fração com exatamente 1):",
          {k: round(float((D[f'n_{k}'] == 1).mean()), 3) for k in UNICOS})
    print("cúspides por tipo (mediana):", D.groupby("tipo").n_cusp.median().to_dict())
    # Conferência de convenção no referencial local (x distal, y apical, z vestibular)
    K_, M_ = np.stack(K), np.stack(M)
    for j, k in enumerate(KP_CLASSES[:6]):
        v = K_[M_[:, j], j]
        print(f"  {k:12s} n {len(v):5d} | x (distal +) {np.median(v[:, 0]):+.2f} | y (apical +) {np.median(v[:, 1]):+.2f} | "
              f"z (vestibular +) {np.median(v[:, 2]):+.2f}")


if __name__ == "__main__":
    main()
