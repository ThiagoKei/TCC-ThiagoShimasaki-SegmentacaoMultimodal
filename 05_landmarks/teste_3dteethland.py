"""Teste dos landmarks (pontos da coroa) no conjunto OFICIAL de teste do 3DTeethLand (MICCAI 2024).

O teste do 3DTeethLand (100 scans: 50 inferiores + 50 superiores) NÃO tem segmentação anotada (o desafio era
totalmente automático), então a avaliação é do SISTEMA: segmentação prevista → recorte por dente → landmarks.
  1. preparo igual ao do treino do 3DTeethLand (01_dados/teeth3ds.py preparar): rotação fixa Teeth3DS → convenção do modelo
     (ROTACAO), recorte de 22 mm a partir das cúspides, decimação para 8.000 células (00_comum/malha.py);
  2. segmentação só 3D: inferior = G4-S150 (o melhor só 3D do TCC), superior = G1-S s42 (o único só 3D nosso da
     superior; ajuste fino do pré-treinado); graph-cut + ilhas com λ = 28 nas duas (λ* escolhido no NOSSO val:
     G4-S150, G1-S superior 53 arcadas do val, 07/11); nada é escolhido no teste;
  3. landmarks so3d (média das 3 sementes; mesmo recorte e referencial do treino);
  4. volta ao referencial do .obj (p_obj = p_prep @ R) e métrica OFICIAL do desafio (evaluation/metrics.py e
     score.py de github.com/crns-smartvision/3DTeethLand, copiados em metricas_3dteethland.py): AP/AR agregando
     todos os scans, 30 limiares de 0 a 2,9 mm, casamento guloso por score; mAP = média das 6 classes.
Também: erro de localização (GT → predição mais próxima da mesma classe), mediana/média e % ≤ 1/2/3 mm.
Uso (precisa do vedo e do gco-wrapper):
  python 05_landmarks/teste_3dteethland.py [--conjunto teste|treino] [--n N] [--saida pasta] [--seg-inferior <best.pt>]
  --conjunto treino é só CONFERÊNCIA DO PIPELINE (o so3d treinou nesses pontos: números otimistas).
"""
import argparse
import json
import shutil
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

AQUI = Path(__file__).resolve().parent
sys.path[:0] = [str(p) for p in sorted(Path(__file__).resolve().parents[1].glob("0*_*"))]  # pastas do repositório
import caminhos  # noqa: E402
import metricas_3dteethland as mt  # noqa: E402
import treinar_fusao as tf  # noqa: E402
from dados_fusao import DatasetFusao, juntar  # noqa: E402
from modelo_landmarks import CANAL_DENTE, ModeloLandmarks, extrair_pontos  # noqa: E402
from landmarks import recortes  # noqa: E402
from recortes_dentes import GRUPO  # noqa: E402
from treinar_landmarks import ESC, UNICOS, condicao  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
T3D = caminhos.TEETH3DS
ROTACAO = {"upper": (-1, -1, -1), "lower": (1, 1, -1)}          # = 01_dados/teeth3ds.py
SENTIDO_COROA = {"upper": -1.0, "lower": 1.0}
ALTURA = 22.0
# Segmentação usada antes dos landmarks (checkpoint, λ do graph-cut). Na monografia: inferior = G4-S150
# (só 3D, do zero, 150 épocas); superior = G1-S (ajuste fino a partir de pesos externos, não reproduzível
# sem eles). Troque com --seg-inferior / --seg-superior.
SEG = {"lower": (caminhos.RAIZ / "runs4" / "G4-S150" / "best.pt", 28.0),
       "upper": (caminhos.RAIZ / "runs_sup" / "G1-S" / "best.pt", 28.0)}
LM = [caminhos.RAIZ / "runs_landmarks" / f"so3d_s{s}" / "best.pt" for s in (42, 43, 44)]
ORDER16 = {"upper": [18, 17, 16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26, 27, 28],
           "lower": [48, 47, 46, 45, 44, 43, 42, 41, 31, 32, 33, 34, 35, 36, 37, 38]}
CLASSES = ["Mesial", "Distal", "Cusp", "InnerPoint", "OuterPoint", "FacialPoint"]
# Cúspides: padrão = limiar 0,3, eps 1,0 mm, score = soma do calor. --cusp-* troca (escolha no val).
CUSP = dict(limiar=0.3, eps=1.0, score="soma")


def matriz(s):
    sx, sy, sz = s
    return np.array([[sx, 0, 0], [0, 0, sy], [0, sz, 0]], dtype=np.float64)


def ler_obj(p):
    v, f = [], []
    with open(p, encoding="utf-8", errors="ignore") as fh:
        for ln in fh:
            if ln.startswith("v "):
                v.append([float(x) for x in ln.split()[1:4]])
            elif ln.startswith("f "):
                f.append([int(x.split("/")[0]) - 1 for x in ln.split()[1:4]])
    return np.asarray(v, np.float64), np.asarray(f, np.int64)


def ler_scan(cid, arch):
    d = T3D / "scans" / arch / cid
    npz = d / f"{cid}_{arch}.npz"
    if npz.exists():
        z = np.load(npz)
        return z["verts"].astype(np.float64), z["faces"].astype(np.int64)
    return ler_obj(d / f"{cid}_{arch}.obj")


def preparar(sc, verts, faces, arch):
    R = matriz(ROTACAO[arch])
    V = verts @ R.T
    h = V[:, 1] * SENTIDO_COROA[arch]
    manter = np.flatnonzero((h[faces] >= np.percentile(h, 99.5) - ALTURA).all(1))
    usados, fc = np.unique(faces[manter], return_inverse=True)
    vd, fd, bd, nd = sc.decimate_to_cells(V[usados], fc.reshape(-1, 3), sc.TARGET_CELLS, label=arch)
    return R, vd, fd, bd, nd


def carregar_gco(sc):
    cut = sc.load_gco()
    if cut is not None:
        return cut
    from gco.pygco import cut_general_graph

    def f(edge_nodes, edge_weights, unaries, pairwise):
        return cut_general_graph(np.ascontiguousarray(edge_nodes, dtype=np.int32),
                                 np.ascontiguousarray(edge_weights, dtype=np.float64),
                                 np.ascontiguousarray(unaries, dtype=np.float64),
                                 np.ascontiguousarray(pairwise, dtype=np.float64), algorithm="expansion")
    return f


def segmentar(modelo, tmp, arch, vd, fd, bd, nd):
    (tmp / "malhas").mkdir(parents=True, exist_ok=True)
    pd.DataFrame([dict(case_id=1, arch=arch, split="teste", arquivo=f"1_{arch}.npz")]).to_csv(tmp / "manifesto.csv",
                                                                                           index=False)
    n = len(fd)
    np.savez(tmp / "malhas" / f"1_{arch}.npz", verts=vd.astype(np.float32), faces=fd.astype(np.int32),
             bary=bd.astype(np.float32), normals=nd.astype(np.float32), label=np.zeros(n, np.int8),
             arco=np.zeros(n, np.float32), rascunho_label=np.zeros(n, np.int8), rascunho_conf=np.zeros(n, np.float16))
    ds = DatasetFusao(tmp, "teste", arch, patch=None, pasta_2d=tmp / "sem_2d")
    b = next(iter(DataLoader(ds, batch_size=1, collate_fn=juntar)))
    with torch.no_grad():
        prob, _ = tf.passo(modelo, b)
    return prob[0].float().numpy().astype(np.float64)


@torch.no_grad()
def landmarks(modelos, rec, arch):
    classes = list(rec)
    if not classes:
        return []
    tipo = np.array([8 - min(c - 1, 16 - c) for c in classes])
    cond = torch.from_numpy(condicao(pd.DataFrame(dict(tipo=tipo, grupo=[GRUPO[t] for t in tipo],
                                                       arch=[arch] * len(classes)))))
    ent = torch.from_numpy(np.stack([rec[c]["pts"] for c in classes])).clone()
    ent[..., :3] /= ESC
    slg, sds = 0, 0
    for m in modelos:
        lg, ds, _ = m(ent, cond)
        slg, sds = slg + lg, sds + ds
    lg, ds = (slg / len(modelos)).numpy(), (sds / len(modelos)).numpy() * ESC
    pontos = []                                     # (classe, xyz no referencial do preparo, score, classe_dente)
    for k, c in enumerate(classes):
        o, R = rec[c]["o"], rec[c]["R"]
        uni, sc, cus = extrair_pontos(rec[c]["pts"][:, :3], lg[k][:, :CANAL_DENTE], ds[k], limiar_cusp=CUSP["limiar"],
                                      eps=CUSP["eps"], score_cusp=CUSP["score"])
        for j, nome in enumerate(UNICOS):
            pontos.append((nome, o + R.T @ uni[j], float(sc[j]), c))
        for p, w in cus:
            s = w if CUSP["score"] == "max" else 1 - np.exp(-w / 5.0)               # score em [0, 1], monotônico
            pontos.append(("Cusp", o + R.T @ p, float(s), c))
    return pontos


def ler_gt(conjunto):
    zp = T3D / f"3DTeethLand_landmarks_{'test' if conjunto == 'teste' else 'train'}.zip"
    gt = {}
    with zipfile.ZipFile(zp) as z:
        for n in z.namelist():
            if n.endswith("__kpt.json"):
                arch, cid = n.split("/")[0], n.split("/")[1]
                gt[(cid, arch)] = json.loads(z.read(n))["objects"]
    return gt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conjunto", choices=["teste", "treino"], default="teste")
    ap.add_argument("--n", type=int, default=None, help="por arcada (conferência rápida)")
    ap.add_argument("--saida", type=Path, default=caminhos.TEETH3DS / "resultado_landmarks")
    ap.add_argument("--cusp-limiar", type=float, default=CUSP["limiar"])
    ap.add_argument("--cusp-eps", type=float, default=CUSP["eps"])
    ap.add_argument("--cusp-score", choices=["soma", "max"], default=CUSP["score"])
    ap.add_argument("--seg-inferior", help="checkpoint da segmentação da arcada inferior")
    ap.add_argument("--seg-superior", help="checkpoint da segmentação da arcada superior")
    a = ap.parse_args()
    CUSP.update(limiar=a.cusp_limiar, eps=a.cusp_eps, score=a.cusp_score)
    print(f"cúspides: {CUSP}", flush=True)
    if a.seg_inferior:
        SEG["lower"] = (Path(a.seg_inferior), SEG["lower"][1])
    if a.seg_superior:
        SEG["upper"] = (Path(a.seg_superior), SEG["upper"][1])
    import malha as sc
    cut = carregar_gco(sc)
    assert cut is not None, "sem gco (pip install gco-wrapper)"
    tf.configurar_precisao("fp32")
    gt = ler_gt(a.conjunto)
    split = "testing" if a.conjunto == "teste" else "training"
    alvo = []
    for arch in ("lower", "upper"):
        ids = [x.rsplit("_", 1)[0] for x in (T3D / "splits" / f"3DTeethLand_{split}_{arch}.txt").read_text().split()]
        alvo += [(c, arch) for c in ids[: a.n] if (c, arch) in gt]
    print(f"{a.conjunto}: {len(alvo)} scans com GT de landmarks", flush=True)
    seg = {}
    for arch, (p, lam) in SEG.items():
        ck = torch.load(p, map_location="cpu", weights_only=False)
        m = tf.novo_modelo(ck)
        m.load_state_dict(ck["model_state_dict"])
        seg[arch] = (m.eval(), lam)
    modelos = []
    for p in LM:
        c = torch.load(p, map_location="cpu", weights_only=False)
        m = ModeloLandmarks(com_2d=False)
        m.load_state_dict(c["modelo"])
        modelos.append(m.eval())
    a.saida.mkdir(parents=True, exist_ok=True)
    tmp = a.saida / "_tmp"
    rng = np.random.default_rng(0)
    pred_rows, por_scan = [], []
    t0 = time.time()
    for i, (cid, arch) in enumerate(alvo):
        verts, faces = ler_scan(cid, arch)
        R, vd, fd, bd, nd = preparar(sc, verts, faces, arch)
        modelo, lam = seg[arch]
        prob = segmentar(modelo, tmp, arch, vd, fd, bd, nd)
        lab, (src, dst) = sc.refine_graphcut(prob, fd.astype(np.int64), bd.astype(float), nd.astype(float), lam, cut)
        lab = sc.remove_islands(lab, src, dst, len(lab), min_frac=0.005, num_classes=17)
        pts = landmarks(modelos, recortes(bd.astype(float), nd.astype(float), lab, arch, rng), arch)
        key = f"{cid}_{arch}"
        for nome, p, s, c in pts:
            q = p @ R                                           # de volta ao referencial do .obj
            pred_rows.append(dict(key=key, coord_x=q[0], coord_y=q[1], coord_z=q[2], **{"class": nome},
                                  score=min(max(s, 0.0), 1.0), fdi=ORDER16[arch][c - 1]))
        n_dentes = len(set(lab[lab > 0].tolist()))
        n_gt = sum(1 for o in gt[(cid, arch)] if o["class"] == "FacialPoint")
        por_scan.append(dict(key=key, arch=arch, dentes_previstos=n_dentes, facial_gt=n_gt, n_pred=len(pts)))
        if (i + 1) % 10 == 0 or i == len(alvo) - 1:
            print(f"[{i + 1}/{len(alvo)}] {(time.time() - t0) / (i + 1):.1f} s/scan", flush=True)
    shutil.rmtree(tmp, ignore_errors=True)
    pred = pd.DataFrame(pred_rows)
    pred.to_csv(a.saida / f"predicoes_{a.conjunto}.csv", index=False)
    pd.DataFrame(por_scan).to_csv(a.saida / f"por_scan_{a.conjunto}.csv", index=False)

    # ---- métrica oficial (score.py) ----
    gold = {k: {} for k in CLASSES}
    for (cid, arch), objs in gt.items():
        if (cid, arch) not in alvo:
            continue
        for o in objs:
            gold[o["class"]].setdefault(f"{cid}_{arch}", []).append(o["coord"])
    resultados = {}
    for nome, filtro in (("todas", None), ("inferior", "lower"), ("superior", "upper")):
        g = {k: {s: v for s, v in d.items() if filtro is None or s.endswith(filtro)} for k, d in gold.items()}
        pa = {k: {} for k in CLASSES}
        for key, x, y, z_, cl, s in zip(pred.key, pred.coord_x, pred.coord_y, pred.coord_z, pred["class"], pred.score):
            if filtro is None or key.endswith(filtro):
                pa[cl].setdefault(key, []).append([[x, y, z_], s])
        resultados[nome] = mt.reformat_scores(mt.score(g, pa))
    # ---- erro de localização: cada ponto GT → predição mais próxima da MESMA classe no mesmo scan ----
    err = {k: [] for k in CLASSES}
    for k in CLASSES:
        for scan, pts_gt in gold[k].items():
            p = pred[(pred.key == scan) & (pred["class"] == k)][["coord_x", "coord_y", "coord_z"]].values
            for q in pts_gt:
                err[k].append(float(np.min(np.linalg.norm(p - np.asarray(q), axis=1))) if len(p) else np.inf)
    loc = {}
    for k, v in err.items():
        v = np.asarray(v)
        f = np.isfinite(v)
        loc[k] = dict(n=len(v), mediana=float(np.median(v[f])), media=float(np.mean(v[f])), sem_pred=int((~f).sum()),
                      ate1=float(np.mean(v <= 1)), ate2=float(np.mean(v <= 2)), ate3=float(np.mean(v <= 3)))
    res = dict(conjunto=a.conjunto, scans=len(alvo), metrica_oficial=resultados, localizacao=loc,
               seg={"lower": "G4-S150 λ=28", "upper": "G1-S s42 λ=28"}, landmarks="so3d s42/43/44 (média)", cuspides=CUSP)
    (a.saida / f"resultado_{a.conjunto}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    for nome, r in resultados.items():
        print(f"{nome}: mAP {r['mAP']:.3f} | mAR {r['mAR']:.3f} | AP cusp {r['AP_cusp']:.3f} M/D "
              f"{r['AP_mesial_distal']:.3f} I/O {r['AP_inner_outer']:.3f} facial {r['AP_facial']:.3f}")
    for k, d in loc.items():
        print(f"  {k:12s} n {d['n']:4d} | mediana {d['mediana']:.2f} mm | média {d['media']:.2f} | ≤1 {100 * d['ate1']:.0f}% "
              f"≤2 {100 * d['ate2']:.0f}% ≤3 {100 * d['ate3']:.0f}% | sem predição {d['sem_pred']}")


if __name__ == "__main__":
    main()
