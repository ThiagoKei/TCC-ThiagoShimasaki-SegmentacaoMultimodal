"""Features 2D por dente para a fusão 2D → 3D.

Roda o Mask R-CNN v2 (output_maskrcnn_flipswap, congelado) com o pipeline NOVO sem linha média — o
mesmo do pré-rótulo v2 — nos casos da coorte 2D com planejamento publicado. Para cada dente que sobra
no pipeline (score >= 0,5, NMS agnóstico, 1 por número, resgate) guarda:
  fdi, classe_modelo (1..32, antes do resgate), score, probs (32 classes), caixa e centróide (imagem 512),
  pos_norm (centróide normalizado ao recorte sem as barras pretas: 0..1), area,
  feat_box  (1024) saída do box_head        -> "features da cabeça de classificação"
  feat_mask (256)  média espacial da entrada do predictor da máscara (28x28) -> forma/aparência
  mask28    (28x28) probabilidade da máscara na classe do modelo, no referencial da caixa
  mask_rle  máscara na imagem 512 (RLE do pycocotools), para conferir com o CVAT depois
e, por imagem, fpn_global (p2..p6 x 256, média espacial de cada nível; para o FiLM) e o recorte.

Saída: <TCC_DADOS>/features_2d_flipswap/<case_id>.npz + indice.csv
Resumível: pula case_id que já tem .npz.

Uso: python 02_detector_2d/extrair_features_2d.py [--n 5] [--conferir] [--espelho] [--extra <pastas>]
  --conferir  compara dentes e números com o pseudo_labels_novo_v2.zip do Kaggle (casos de teste)
  --espelho   panorâmica invertida na horizontal (augmentation "espelho dos dois exames") →
              <TCC_DADOS>/features_2d_flipswap_espelho/; no fim compara com a extração original
              (mesmos dentes com o lado trocado, x espelhado, semelhança das features)
"""
import argparse
import sys
import time
from pathlib import Path

AQUI = Path(__file__).resolve().parent
sys.path[:0] = [str(p) for p in sorted(Path(__file__).resolve().parents[1].glob("0*_*"))]  # pastas do repositório
import caminhos  # noqa: E402
import pipeline_2d as p2d  # noqa: E402
from montar_cvat_teste_multimodal import BASE, jsons_do_kaggle  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PESOS = caminhos.PESOS / "maskrcnn_d2_flipswap.pth"     # detector D2 (o usado na fusão)
JSONS_V2 = caminhos.PESOS / "pseudo_labels_novo_v2.zip"   # pré-rótulos v2 gerados no Kaggle (tcc-xray-v2.ipynb)
SAIDA = caminhos.FEATURES_2D
SAIDA_ESPELHO = caminhos.FEATURES_2D_ESPELHO
CORTE = pd.Timestamp("2026-08-14")                     # data dos pesos pré-treinados: casos posteriores nunca foram vistos por eles


def universo(extras=()):
    """Casos da coorte 2D (ds_limpo/*_limpo) com planejamento publicado, novos e antigos,
    menos os marcados como pulado_* na coluna status do teste_multimodal_ordem.csv.
    extra: pasta adicional de panorâmicas já revisadas (ds_complementar/processados, os casos do
    treino do MeshSegNet pré-treinado), fora do ds_limpo para não mudar a contagem oficial da coorte."""
    z = pd.read_csv(caminhos.LISTAS / "casos_planejamento.csv", dtype={"case_id": str})
    z["primeira_publicacao"] = pd.to_datetime(z["primeira_publicacao"])
    zc = z.groupby("case_id")["primeira_publicacao"].min().reset_index()
    exts = {".png", ".jpg", ".jpeg"}
    pastas = sorted((BASE / "ds_limpo").glob("*_limpo")) + [Path(e) for e in extras]
    c = pd.DataFrame([(p.stem.split(".")[0], p.relative_to(BASE).as_posix())
                      for d in pastas
                      for p in sorted(d.rglob("*")) if p.suffix.lower() in exts],
                     columns=["case_id", "panoramica"]).drop_duplicates("case_id")
    m = c.merge(zc, on="case_id")
    m["novo"] = m["primeira_publicacao"] >= CORTE
    ordem = pd.read_csv(caminhos.LISTAS / "teste_multimodal_ordem.csv", dtype={"case_id": str, "status": str})
    pulados = set(ordem.case_id[ordem.status.fillna("").str.startswith("pulado")])
    m = m[~m.case_id.isin(pulados)].merge(ordem[["case_id", "ordem"]], on="case_id", how="left")
    return m.sort_values(["ordem", "case_id"], na_position="last").reset_index(drop=True)


def conferir(casos):
    """Compara o que foi extraído com o pré-rótulo v2 do Kaggle (mesmo modelo e pipeline, em GPU)."""
    kaggle = jsons_do_kaggle(JSONS_V2)
    linhas = []
    for c in casos:
        f = SAIDA / f"{c}.npz"
        if c not in kaggle or not f.exists():
            continue
        loc = set(np.load(f)["classe"].tolist())
        kag = {a["category_id"] for a in kaggle[c]["annotations"]}
        linhas.append((c, len(loc), len(kag), len(loc & kag) / max(len(loc | kag), 1)))
    d = pd.DataFrame(linhas, columns=["case_id", "dentes_local", "dentes_kaggle", "jaccard_numeros"])
    print(f"\nConferência com {JSONS_V2.name} ({len(d)} casos): Jaccard dos números "
          f"médio {d.jaccard_numeros.mean():.3f}, idêntico em {(d.jaccard_numeros == 1).mean():.0%}; "
          f"dentes local {d.dentes_local.sum()} × Kaggle {d.dentes_kaggle.sum()}")
    print(d.sort_values("jaccard_numeros").head(5).to_string(index=False))


def espelhar_fdi(f):
    return {1: 2, 2: 1, 3: 4, 4: 3}[f // 10] * 10 + f % 10


def comparar_espelho(casos):
    """Original × panorâmica invertida, por caso: os dentes casados pelo x espelhado (pos_norm x → 1 − x,
    mesma arcada, gulosamente pelo mais próximo). Conta quantos casam, se o número do espelho é o do
    original com o lado trocado, e o cosseno do feat_box entre o dente e o seu espelho."""
    linhas = []
    for c in casos:
        fo, fe = SAIDA / f"{c}.npz", SAIDA_ESPELHO / f"{c}.npz"
        if not fo.exists() or not fe.exists():
            continue
        o, e = np.load(fo), np.load(fe)
        livres, n_ok, cos, dx = set(range(len(e["fdi"]))), 0, [], []
        for i in range(len(o["fdi"])):
            cand = [j for j in livres if e["arcada"][j] == o["arcada"][i]]
            if not cand:
                continue
            d = [abs((1 - e["pos_norm"][j, 0]) - o["pos_norm"][i, 0]) for j in cand]
            j = cand[int(np.argmin(d))]
            if min(d) > 0.03:                    # ~meio dente na largura do recorte
                continue
            livres.discard(j)
            n_ok += int(e["fdi"][j]) == espelhar_fdi(int(o["fdi"][i]))
            a, b = o["feat_box"][i].astype(np.float32), e["feat_box"][j].astype(np.float32)
            cos.append(float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)))
            dx.append(min(d))
        linhas.append(dict(case_id=c, dentes=len(o["fdi"]), dentes_espelho=len(e["fdi"]), casados=len(cos),
                           numero_espelhado_ok=n_ok, cos_feat_box=np.mean(cos) if cos else np.nan,
                           dx_mediano=np.median(dx) if dx else np.nan))
    d = pd.DataFrame(linhas)
    print(f"\nOriginal × espelho ({len(d)} casos): dentes {d.dentes.sum()} × {d.dentes_espelho.sum()}; "
          f"casados pelo x espelhado {d.casados.sum()}; destes, número = original com o lado trocado em "
          f"{d.numero_espelhado_ok.sum() / max(d.casados.sum(), 1):.1%}")
    print(f"cosseno médio do feat_box (dente × espelho): {d.cos_feat_box.mean():.3f}; "
          f"|dx| mediano {d.dx_mediano.median():.4f} da largura do recorte")
    print(d.sort_values("numero_espelhado_ok").head(8).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, help="só os n primeiros (teste)")
    ap.add_argument("--conferir", action="store_true")
    ap.add_argument("--espelho", action="store_true", help="panorâmica invertida → features_2d_flipswap_espelho")
    ap.add_argument("--extra", type=Path, nargs="*", default=[],
                    help="pastas extras de panorâmicas (ds_complementar/processados; também "
                         "ds_ampliacao/processados). Passe TODAS: o indice.csv é refeito com o universo da rodada")
    ap.add_argument("--dispositivo", default="cpu", help="cpu (como no TCC) ou cuda")
    args = ap.parse_args()
    saida = SAIDA_ESPELHO if args.espelho else SAIDA

    saida.mkdir(parents=True, exist_ok=True)
    u = universo([e if e.is_absolute() else BASE / e for e in args.extra])
    if args.n:
        u = u.head(args.n)
    print(f"{len(u)} casos ({u.novo.sum()} novos, {(~u.novo).sum()} antigos); pesos {PESOS}")

    pred = p2d.build_predictor(PESOS, args.dispositivo)
    cap = p2d.instalar_captura(pred)
    t0, feitos = time.time(), 0
    for i, r in enumerate(u.itertuples(), 1):
        f = saida / f"{r.case_id}.npz"
        if f.exists():
            continue
        img = p2d.imread(BASE / r.panoramica)
        d, _ = p2d.extrair(pred, cap, img[:, ::-1].copy() if args.espelho else img)
        np.savez_compressed(f, **d)
        feitos += 1
        rest = (time.time() - t0) / feitos * (len(u) - i)
        print(f"[{i:>3}/{len(u)}] {r.case_id}: {len(d['fdi'])} dentes  (faltam ~{rest / 60:.0f} min)", flush=True)

    idx = []
    for r in u.itertuples():
        d = np.load(saida / f"{r.case_id}.npz")
        idx.append({"case_id": r.case_id, "ordem": r.ordem, "novo": r.novo, "panoramica": r.panoramica,
                    "n_dentes": len(d["fdi"]), "fdi": " ".join(map(str, sorted(d["fdi"].tolist())))})
    pd.DataFrame(idx).to_csv(saida / "indice.csv", index=False)
    print(f"\n{len(idx)} casos em {saida} (mediana {np.median([x['n_dentes'] for x in idx]):.0f} dentes/imagem)")
    if args.conferir and not args.espelho:
        conferir(u.case_id[u.ordem <= 150])
    if args.espelho:
        comparar_espelho(u.case_id)


if __name__ == "__main__":
    main()
