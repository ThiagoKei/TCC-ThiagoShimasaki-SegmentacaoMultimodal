"""Métricas do experimento final (G5) no teste, a partir do que o rodar_g5.sh grava.

Subcomandos:
  mcnemar [gc|cru]   regra pré-registrada: FR.sN × SC.sN por semente, McNemar de dentes certos (DSC >= 0,5), com
                     graph-cut no λ* do val (gc, principal; inclui também o cru) ou só cru. Secundárias: DSC dos
                     dentes, DSC17, dentes inventados, arcadas 100 % certas; subconjuntos 418 = 139 antigos + 279 novos;
                     ablação FR sem a panorâmica.
  por-tipo [gc|cru]  o mesmo por tipo de dente (1 = incisivo central … 8 = 3º molar) e o McNemar sem os 3º molares.
  miou               mIoU a partir do DSC por classe já salvo (IoU = DSC / (2 − DSC)) → metricas_extra/miou_g5.csv.
  acuracia           acurácia por célula e IoU exato com o graph-cut no λ* (precisa do gco-wrapper)
                     → metricas_extra/acuracia_gc.csv.
  tudo               todos acima (como no fim do rodar_g5.sh).

Entradas: <TCC_DADOS>/runs5/<run>/teste_por_classe.csv (cru), <TCC_DADOS>/g5_resultados/lambda/<run>/
teste_por_classe_gc.csv (graph-cut), <TCC_DADOS>/g5_resultados/probs/ e o manifesto.csv (os 139 casos antigos).
Uso: python 06_avaliacao/avaliar_g5.py {mcnemar,por-tipo,miou,acuracia,tudo} [gc|cru]
"""
import csv
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
sys.path[:0] = [str(p) for p in sorted(Path(__file__).resolve().parents[1].glob("0*_*"))]  # pastas do repositório
import caminhos  # noqa: E402

CRU, GC = caminhos.RUNS, caminhos.RESULTADOS / "lambda"
SEMENTES = ["s42", "s43", "s44"]


def antigos():
    """Os 139 casos do teste original (conjunto multimodal anotado)."""
    m = pd.read_csv(caminhos.FUSAO_DADOS / "manifesto.csv")
    return set(m[(m.split == "teste") & (m.arch == "lower")].case_id)


def p_mcnemar(b, c):
    """McNemar exato (binomial bilateral) com b e c discordantes."""
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


# ── mcnemar ─────────────────────────────────────────────────────────────────────────────────────────────────────────
def _ler(run, gc, sem_2d=False):
    f = GC / run / "teste_por_classe_gc.csv" if gc else CRU / run / "teste_por_classe.csv"
    if not f.exists():
        return None
    t = pd.read_csv(f)
    return t[t.sem_2d == sem_2d]


def _pres(t):
    return t[(t.classe > 0) & t.no_gt].set_index(["case_id", "classe"]).dsc


def _comparar(base, fus, sub=None):
    j = pd.concat([_pres(base).rename("b"), _pres(fus).rename("f")], axis=1).dropna()
    if sub is not None:
        j = j[j.index.get_level_values(0).isin(sub)]
    cb, cf = j.b >= 0.5, j.f >= 0.5
    c, q = int((~cb & cf).sum()), int((cb & ~cf).sum())
    return c, q, p_mcnemar(c, q), len(j)


def _secund(t):
    dentes = t[(t.classe > 0) & t.no_gt].groupby("case_id").dsc.mean().mean()
    d17 = t.groupby("case_id").dsc.mean().mean()
    inv = t[(t.classe > 0) & ~t.no_gt & (t.dsc < 0.5)]
    inv3 = inv[inv.fdi.isin([38, 48])]
    p = t[(t.classe > 0) & t.no_gt]
    arc = (p.dsc >= 0.5).groupby(p.case_id).all()
    return dict(dentes=dentes, d17=d17, inventados=len(inv), inv_3M=len(inv3), arc100=int(arc.sum()), n=len(arc))


def mcnemar(modo="gc"):
    ant = antigos()
    for gc in ([True, False] if modo == "gc" else [False]):
        print(f"\n===== {'GRAPH-CUT λ*' if gc else 'CRU'} =====")
        print("McNemar FR × SC por semente (corrigidos / quebrados, p) — 418 | 139 antigos | 279 novos")
        for s in SEMENTES:
            fr, sc = _ler(f"G5-FR.{s}", gc), _ler(f"G5-SC.{s}", gc)
            if fr is None or sc is None:
                print(f"  {s}: falta run"); continue
            out = []
            for sub in [None, ant, set(fr.case_id) - ant]:
                c, q, p, n = _comparar(sc, fr, sub)
                out.append(f"{c}/{q} p={p:.3f}")
            print(f"  {s}: " + " | ".join(out))
        print("secundárias (418):")
        for r in ["G5-S", "G5-SC", "G5-FR"]:
            for s in SEMENTES:
                t = _ler(f"{r}.{s}", gc)
                if t is not None:
                    d = _secund(t)
                    print(f"  {r}.{s:4s} dentes {d['dentes']:.4f} | DSC17 {d['d17']:.4f} | inventados {d['inventados']} "
                          f"(3º M {d['inv_3M']}) | arcadas 100 % {d['arc100']}/{d['n']}")
    print("\n===== ABLAÇÃO FR sem 2D → com 2D (só cru: a fase D não gravou probs sem 2D) =====")
    for s in SEMENTES:
        com, sem = _ler(f"G5-FR.{s}", False), _ler(f"G5-FR.{s}", False, sem_2d=True)
        c, q, p, n = _comparar(sem, com)
        ds = _secund(sem)
        print(f"  {s}: corrigidos {c} / quebrados {q} p={p:.3f} | sem 2D: inventados {ds['inventados']} (3º M {ds['inv_3M']})")


# ── por-tipo ────────────────────────────────────────────────────────────────────────────────────────────────────────
def por_tipo(modo="cru"):
    gc = modo == "gc"
    raiz = GC if gc else CRU
    arq = "teste_por_classe_gc.csv" if gc else "teste_por_classe.csv"

    def ler(r, s2=False):
        f = raiz / r / arq
        if not f.exists():
            return None
        t = pd.read_csv(f)
        t = t[t.sem_2d == s2]
        return t if len(t) else None

    print("GRAPH-CUT λ*" if gc else "CRU")
    for s in SEMENTES:
        sc, fr, fr0 = ler(f"G5-SC.{s}"), ler(f"G5-FR.{s}"), ler(f"G5-FR.{s}", True)
        if sc is None or fr is None:
            continue
        rows = []
        for nome, t in [("SC", sc), ("FR", fr), ("FR sem2D", fr0)]:
            if t is None:
                continue
            t = t[t.classe > 0].copy()
            t["tipo"] = t.fdi % 10
            p, a = t[t.no_gt], t[~t.no_gt]
            rows.append(pd.Series({f"{k}": f"{(g.dsc >= .5).sum()}/{len(g)}" for k, g in p.groupby("tipo")},
                                  name=f"{nome} certos/GT"))
            rows.append(pd.Series({f"{k}": f"{(g.dsc < .5).sum()}" for k, g in a.groupby("tipo")}, name=f"{nome} inventados"))
        j = pd.concat([sc[(sc.classe > 0) & sc.no_gt].set_index(["case_id", "fdi"]).dsc.rename("b"),
                       fr[(fr.classe > 0) & fr.no_gt].set_index(["case_id", "fdi"]).dsc.rename("f")], axis=1)
        j["tipo"] = j.index.get_level_values(1) % 10
        j["corr"] = (j.b < .5) & (j.f >= .5)
        j["queb"] = (j.b >= .5) & (j.f < .5)
        print(f"\n=== {s} (tipo 1 = incisivo central … 8 = 3º molar) ===")
        print(pd.DataFrame(rows).to_string())
        print(j.groupby("tipo")[["corr", "queb"]].sum().T.to_string())
        s7 = j[j.tipo < 8]
        c, q = int(s7["corr"].sum()), int(s7["queb"].sum())
        print(f"dentes 1–7: corrigidos {c} / quebrados {q}, McNemar p = {p_mcnemar(c, q):.3f}")


# ── miou ────────────────────────────────────────────────────────────────────────────────────────────────────────────
def _metricas(t):
    t = t[t.no_gt].copy()
    t["iou"] = t.dsc / (2 - t.dsc)
    por = t.groupby("case_id").agg(dsc_todas=("dsc", "mean"), miou_todas=("iou", "mean"))
    d = t[t.classe > 0].groupby("case_id").agg(dsc_dentes=("dsc", "mean"), miou_dentes=("iou", "mean"))
    return por.join(d)


def miou():
    ant = antigos()
    linhas = []
    for modelo in ["G5-S", "G5-SC", "G5-FR"]:
        for s in [42, 43, 44]:
            run = f"{modelo}.s{s}"
            fontes = {"cru": CRU / run / "teste_por_classe.csv", "gc": GC / run / "teste_por_classe_gc.csv"}
            for etapa, arq in fontes.items():
                if not arq.exists():
                    print("falta", arq); continue
                t = pd.read_csv(arq)
                variantes = [("com 2D" if "FR" in modelo else "-", False)]
                if "FR" in modelo and etapa == "cru":
                    variantes.append(("sem 2D", True))
                for rot, sem in variantes:
                    m = _metricas(t[t.sem_2d == sem])
                    for sub, mm in [("418", m), ("139", m[m.index.isin(ant)]), ("279", m[~m.index.isin(ant)])]:
                        linhas.append(dict(modelo=modelo, semente=s, radiografia=rot, etapa=etapa, subconj=sub,
                                           n=len(mm), **mm.mean().round(4).to_dict()))
    D = pd.DataFrame(linhas)
    out = caminhos.RESULTADOS / "metricas_extra" / "miou_g5.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    D.to_csv(out, index=False)
    pd.set_option("display.width", 200)
    print(D[D.subconj == "418"].to_string(index=False))
    print()
    med = D.groupby(["modelo", "radiografia", "etapa", "subconj"])[["dsc_todas", "miou_todas", "dsc_dentes", "miou_dentes"]]
    print(med.mean().round(4).to_string())


# ── acuracia ────────────────────────────────────────────────────────────────────────────────────────────────────────
def _iou_presentes(pred, gt):
    ious, ious_d = [], []
    for c in range(17):
        g = gt == c
        if not g.any():
            continue
        p = pred == c
        v = (p & g).sum() / max((p | g).sum(), 1)
        ious.append(v)
        if c > 0:
            ious_d.append(v)
    return float(np.mean(ious)), float(np.mean(ious_d))


def acuracia():
    import malha as sc
    malhas, probs = caminhos.FUSAO_DADOS / "malhas", caminhos.RESULTADOS / "probs"
    saida = caminhos.RESULTADOS / "metricas_extra" / "acuracia_gc.csv"
    saida.parent.mkdir(parents=True, exist_ok=True)
    runs = [f"{m}.s{s}" for m in ["G5-S", "G5-SC", "G5-FR"] for s in [42, 43, 44]]
    cut = sc.load_gco()
    assert cut is not None, "sem gco"
    min_frac = 0.005
    lam = {r["run"]: float(r["lambda_val"]) for r in csv.DictReader(open(GC / "lambda_resumo.csv", encoding="utf-8"))}
    linhas, t0 = [], time.time()
    for run in runs:
        arqs = [f for f in sorted((probs / run).glob("*.npz")) if str(np.load(f)["split"]) == "teste"]
        for f in arqs:
            prob = np.load(f)["prob"].astype(np.float64)
            z = np.load(malhas / f.name)
            gt = z["label"].astype(int)
            cru = prob.argmax(1)
            lab, (src, dst) = sc.refine_graphcut(prob, z["faces"].astype(np.int64), z["bary"].astype(float),
                                                 z["normals"].astype(float), lam[run], cut)
            gc = sc.remove_islands(lab, src, dst, len(lab), min_frac=min_frac, num_classes=17)
            mi, mid = _iou_presentes(gc, gt)
            linhas.append(dict(run=run, case_id=int(f.stem.split("_")[0]), lam=lam[run],
                               acc_cru=float((cru == gt).mean()), acc_gc=float((gc == gt).mean()),
                               miou_gc_exato=mi, miou_dentes_gc_exato=mid))
        print(f"{run}: {len(arqs)} arcadas | λ={lam[run]} | {time.time() - t0:.0f} s", flush=True)
        with open(saida, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(linhas[0]))
            w.writeheader()
            w.writerows(linhas)
    print("ok", saida)


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    modo = sys.argv[2] if len(sys.argv) > 2 else None
    if cmd == "mcnemar":
        mcnemar(modo or "gc")
    elif cmd == "por-tipo":
        por_tipo(modo or "cru")
    elif cmd == "miou":
        miou()
    elif cmd == "acuracia":
        acuracia()
    elif cmd == "tudo":
        mcnemar("gc"); mcnemar("cru"); por_tipo("gc"); miou(); acuracia()
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
