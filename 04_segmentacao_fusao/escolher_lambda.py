"""Graph-cut (00_comum/malha.py) com λ escolhido no VAL, por modelo. Precisa do gco-wrapper.
  python 04_segmentacao_fusao/escolher_lambda.py --probs <TCC_DADOS>/g5_resultados/probs --saida <TCC_DADOS>/g5_resultados/lambda

Para cada run em --probs (uma subpasta por run; na G5: G5-S, G5-SC e G5-FR de cada semente e as dobras G5-S.k):
  1. val: DSC dos dentes com o argmax cru e com graph-cut (malha.refine_graphcut) + remove_islands para cada λ
     da grade → λ* = o melhor no val (o teste não participa da escolha);
  2. teste: aplica cru, λ* e o λ = 40 de referência e grava, ao lado do teste_por_classe.csv original, o
     teste_por_classe_gc.csv (mesmo formato, coluna `sem_2d` = False) em <saida>/<run>/, lido pelos scripts de 06_avaliacao.
As malhas (faces, baricentros, normais, rótulo GT) vêm de <TCC_DADOS>/fusao_dados/malhas (a geometria é a mesma
nos dados limpos). Resumo em <saida>/lambda_resumo.csv.
"""
import argparse
import sys
from pathlib import Path

import csv

import numpy as np

sys.path[:0] = [str(p) for p in sorted(Path(__file__).resolve().parents[1].glob("0*_*"))]  # pastas do repositório
import caminhos  # noqa: E402
import malha as sc  # noqa: E402

MALHAS = caminhos.FUSAO_DADOS / "malhas"
ORDER16 = [48, 47, 46, 45, 44, 43, 42, 41, 31, 32, 33, 34, 35, 36, 37, 38]
GRADE = [0.5, 1, 2, 3, 5, 7, 10, 14, 20, 28, 40]


def dsc_por_classe(pred, labels, n=17, smooth=1.0):          # igual ao treinar_fusao.py
    d, pres = np.zeros(n), np.zeros(n, bool)
    for c in range(n):
        p, g = pred == c, labels == c
        pres[c] = g.any()
        d[c] = (2.0 * (p & g).sum() + smooth) / (p.sum() + g.sum() + smooth)
    return d, pres


def dsc_dentes(pred, gt):
    d, pres = dsc_por_classe(pred, gt)
    return float(d[1:][pres[1:]].mean())


def escrever(arq, linhas):                                  # sem pandas
    if not linhas:
        return
    with open(arq, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(linhas[0]))
        w.writeheader()
        w.writerows(linhas)


def main():
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--probs", type=Path, required=True)
    ap.add_argument("--saida", type=Path, default=None, help="padrão: a própria pasta --probs")
    ap.add_argument("--min-frac", type=float, default=None, help="remove_islands; padrão 0,005")
    ap.add_argument("--retomar", action="store_true",
                    help="pula os runs que já estão no lambda_resumo.csv; o resumo é regravado após cada run")
    a = ap.parse_args()
    saida = a.saida or a.probs
    cut = sc.load_gco()
    assert cut is not None, "sem gco: pip install gco-wrapper"
    min_frac = a.min_frac if a.min_frac is not None else 0.005

    def refinar(prob, z, lam):
        lab, (src, dst) = sc.refine_graphcut(prob, z["faces"].astype(np.int64), z["bary"].astype(float),
                                             z["normals"].astype(float), lam, cut)
        return sc.remove_islands(lab, src, dst, len(lab), min_frac=min_frac, num_classes=17)

    resumo = []
    arq_resumo = saida / "lambda_resumo.csv"
    if a.retomar and arq_resumo.exists():                   # runs já feitos ficam no resumo e são pulados
        with open(arq_resumo, encoding="utf-8") as fh:
            resumo = list(csv.DictReader(fh))
    feitos = {r["run"] for r in resumo}
    for run in sorted(p for p in a.probs.iterdir() if p.is_dir()):
        if run.name in feitos:
            print(f"{run.name}: já no lambda_resumo.csv, pulando")
            continue
        arqs = sorted(run.glob("*.npz"))
        por_split = {"val": [], "teste": []}
        for f in arqs:
            por_split[str(np.load(f)["split"])].append(f)
        if not por_split["val"]:
            print(f"{run.name}: sem arcadas de val, pulando")
            continue
        # ---- 1. λ no val ----
        val = {lam: [] for lam in ["cru"] + GRADE}
        for f in por_split["val"]:
            prob = np.load(f)["prob"].astype(np.float64)
            z = np.load(MALHAS / f.name)
            gt = z["label"].astype(int)
            val["cru"].append(dsc_dentes(prob.argmax(1), gt))
            for lam in GRADE:
                val[lam].append(dsc_dentes(refinar(prob, z, lam), gt))
        med = {k: float(np.mean(v)) for k, v in val.items()}
        lam_star = max(GRADE, key=lambda l: med[l])
        print(f"{run.name} val ({len(por_split['val'])} arcadas): " +
              " | ".join(f"{k}: {v:.4f}" for k, v in med.items()) + f"  →  λ* = {lam_star}")
        # ---- 2. teste: cru, λ*, λ = 40 ----
        linhas, dte = [], {"cru": [], "lam*": [], "lam40": []}
        for f in por_split["teste"]:
            prob = np.load(f)["prob"].astype(np.float64)
            z = np.load(MALHAS / f.name)
            gt = z["label"].astype(int)
            cid = int(f.stem.split("_")[0])
            for nome, pred in (("cru", prob.argmax(1)), ("lam*", refinar(prob, z, lam_star)), ("lam40", refinar(prob, z, 40.0))):
                d, pres = dsc_por_classe(pred, gt)
                dte[nome].append(float(d[1:][pres[1:]].mean()))
                if nome == "lam*":
                    linhas += [dict(case_id=cid, classe=c, fdi=([0] + ORDER16)[c], dsc=d[c], no_gt=pres[c], sem_2d=False)
                               for c in range(17)]
        (saida / run.name).mkdir(parents=True, exist_ok=True)
        escrever(saida / run.name / "teste_por_classe_gc.csv", linhas)
        r = dict(run=run.name, lambda_val=lam_star, val_cru=med["cru"], val_lam=med[lam_star], val_lam40=med[40],
                 teste_cru=np.mean(dte["cru"]), teste_lam=np.mean(dte["lam*"]), teste_lam40=np.mean(dte["lam40"]),
                 n_val=len(por_split["val"]), n_teste=len(por_split["teste"]))
        resumo.append(r)
        print(f"   teste ({r['n_teste']}): cru {r['teste_cru']:.4f} | λ*={lam_star} {r['teste_lam']:.4f} | λ=40 {r['teste_lam40']:.4f}")
        escrever(arq_resumo, resumo)                         # a cada run: uma queda não perde o que já saiu
    escrever(arq_resumo, resumo)


if __name__ == "__main__":
    main()
