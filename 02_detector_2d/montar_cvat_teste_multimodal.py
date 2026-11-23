"""Monta a pasta do conjunto de teste multimodal: pacote do CVAT (panorâmicas + pré-rótulo) e os .vtp.

Pré-rótulo = pipeline NOVO (pipeline_2d.py, as mesmas funções da célula do tcc-xray-v2.ipynb) com o modelo
atual, em CPU. É o mesmo que a fase 1 do notebook gera no Kaggle.

Saída em <TCC_DADOS>/cvat_teste_multimodal (os .vtp são geometria de paciente):
  cvat_teste_multimodal.zip   -> CVAT: Create task > Upload dataset/annotations, formato "COCO 1.0"
      images/<ordem>_<case_id>.png
      annotations/instances_default.json   (category "FDI-xx", polígonos, coordenadas da imagem 512)
  3d/{upper,lower}/<ordem>_<case_id>_<arch>.vtp   (hard links para <TCC_BRUTOS>/tcc_*: 0 byte extra)
  manifest.csv   (ordem, case_id, panorâmica de origem, rótulo do ciclo 3D, status_3d)

Uso: python 02_detector_2d/montar_cvat_teste_multimodal.py [--n 150] [--tag v2]
  v2 a partir do Kaggle:  ... --tag v2 --jsons <Downloads>/pseudo_labels_novo_v2.zip
  v2 inferindo aqui:      ... --tag v2 --pesos <...>/output_maskrcnn_flipswap/model_final.pth --sem-linha-media
"""
import argparse
import json
import os
import shutil
import sys
import zipfile
from pathlib import Path

AQUI = Path(__file__).resolve().parent
sys.path[:0] = [str(p) for p in sorted(Path(__file__).resolve().parents[1].glob("0*_*"))]  # pastas do repositório
import caminhos  # noqa: E402

BASE = caminhos.PANORAMICAS                             # pastas de panorâmicas pré-processadas

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import pipeline_2d as p2d  # noqa: E402

PESOS = caminhos.PESOS / "maskrcnn_d1.pth"              # detector D1 (pré-rótulo v1)
VTP = caminhos.BRUTOS
SAIDA = caminhos.CVAT_TESTE


def jsons_do_kaggle(origem):
    """{case_id: anotações} a partir do pseudo_labels_novo_v2 (zip ou pasta) gerado pelo Kaggle.
    O case_id sai do nome da panorâmica (<case_id>.0.png), que não depende da estrutura de pastas."""
    origem = Path(origem)
    out = {}
    if origem.suffix == ".zip":
        with zipfile.ZipFile(origem) as z:
            for n in z.namelist():
                if n.endswith(".json"):
                    d = json.loads(z.read(n))
                    out[Path(d["file_name"]).name.split(".")[0]] = d
    else:
        for p in origem.rglob("*.json"):
            d = json.loads(p.read_text())
            out[Path(d["file_name"]).name.split(".")[0]] = d
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--jsons", help="pseudo_labels_novo_v2.zip (ou pasta) do Kaggle: usa esses pré-rótulos, sem inferir")
    ap.add_argument("--pesos", default=str(PESOS), help="pesos para inferir aqui (ignorado com --jsons)")
    ap.add_argument("--sem-linha-media", action="store_true", help="desliga a correção de lado (modelo retreinado)")
    ap.add_argument("--tag", default="", help="sufixo da saída, ex.: v2 -> cvat_v2/, manifest_v2.csv, *_v2.zip")
    args = ap.parse_args()
    suf = f"_{args.tag}" if args.tag else ""

    ordem = pd.read_csv(caminhos.LISTAS / "teste_multimodal_ordem.csv", dtype={"case_id": str}).head(args.n)
    cvat_dir = SAIDA / f"cvat{suf}"
    img_dir = cvat_dir / "images"
    ann_dir = cvat_dir / "annotations"
    for d in (img_dir, ann_dir, SAIDA / "3d" / "upper", SAIDA / "3d" / "lower"):
        d.mkdir(parents=True, exist_ok=True)

    if args.jsons:
        kaggle = jsons_do_kaggle(args.jsons)
        print(f"{len(kaggle)} pré-rótulos lidos de {args.jsons}")
        faltam = [c for c in ordem.case_id if c not in kaggle]
        if faltam:
            raise SystemExit(f"{len(faltam)} casos sem JSON no {args.jsons}: {faltam[:10]}")
        prerrotulo = lambda img, case_id: kaggle[case_id]["annotations"]
    else:
        pred = p2d.build_predictor(args.pesos, "cpu")
        prerrotulo = lambda img, case_id: p2d.to_annotations(
            p2d.predict_novo(pred, img, not args.sem_linha_media))
    fdi = {q * 8 + t + 1: (q + 1) * 10 + t + 1 for q in range(4) for t in range(8)}
    categories = [{"id": c, "name": f"FDI-{fdi[c]}", "supercategory": "tooth"} for c in range(1, 33)]

    images, annotations, manifest = [], [], []
    for r in ordem.itertuples():
        nome = f"{int(r.ordem):03d}_{r.case_id}.png"
        src = BASE / r.panoramica
        shutil.copy2(src, img_dir / nome)
        img = p2d.imread(src)
        H, W = img.shape[:2]
        img_id = len(images) + 1
        images.append({"id": img_id, "file_name": nome, "width": W, "height": H})
        anns = prerrotulo(img, r.case_id)
        if args.jsons and kaggle[r.case_id]["height"] != H:
            raise SystemExit(f"{r.case_id}: o JSON do Kaggle é de uma imagem {kaggle[r.case_id]['height']}px, a local tem {H}px")
        for a in anns:
            x, y, w, h = a["bbox"]
            annotations.append({"id": len(annotations) + 1, "image_id": img_id, "category_id": a["category_id"],
                                "bbox": [x, y, w, h], "area": float(w * h), "iscrowd": 0,
                                "segmentation": [s for s in a["segmentation"] if len(s) >= 6],
                                "attributes": {"score": round(a["score"], 3)}})

        rotulos = sorted({p.name.rsplit("_", 1)[0] for arch in ("upper", "lower")
                          for p in (VTP / f"tcc_{arch}").glob(f"*_{r.case_id}_v*_{arch}.vtp")})
        for rot in rotulos[-1:]:                     # ciclo mais recente, se houver mais de um
            for arch in ("upper", "lower"):
                v = VTP / f"tcc_{arch}" / f"{rot}_{arch}.vtp"
                dst = SAIDA / "3d" / arch / f"{int(r.ordem):03d}_{r.case_id}_{arch}.vtp"
                if v.exists() and not dst.exists():
                    try:
                        os.link(v, dst)
                    except OSError:
                        shutil.copy2(v, dst)
        manifest.append({"ordem": r.ordem, "case_id": r.case_id, "imagem_cvat": nome, "panoramica": r.panoramica,
                         "ciclo_3d": rotulos[-1] if rotulos else "", "status_3d": r.status_3d,
                         "dentes_prerrotulo": len(anns)})
        print(f"[{int(r.ordem):>3}] {r.case_id}: {len(anns)} dentes pré-rotulados", flush=True)

    with open(ann_dir / "instances_default.json", "w", encoding="utf-8") as f:
        json.dump({"images": images, "annotations": annotations, "categories": categories}, f)
    pd.DataFrame(manifest).to_csv(SAIDA / f"manifest{suf}.csv", index=False)

    zip_path = SAIDA / f"cvat_teste_multimodal{suf}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(cvat_dir.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(cvat_dir).as_posix())
    n_vtp = sum(1 for _ in (SAIDA / "3d").rglob("*.vtp"))
    print(f"\n{len(images)} imagens, {len(annotations)} dentes pré-rotulados "
          f"(mediana {np.median([m['dentes_prerrotulo'] for m in manifest]):.0f}/imagem), {n_vtp} .vtp")
    print(f"Pacote do CVAT: {zip_path} ({zip_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
