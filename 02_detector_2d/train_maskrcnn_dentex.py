"""
train_maskrcnn_dentex.py
========================
Estágios:
  prep      — converte DENTEX para COCO + aplica CLAHE nas imagens
  train     — treina Mask R-CNN (32 classes de dente) no DENTEX
  prelabel  — gera pseudo-labels (Mask R-CNN → SAM2) para o dataset privado 7k

Uso típico no Kaggle (execute um estágio por run, salvando outputs entre eles):
  python train_maskrcnn_dentex.py --stage prep
  python train_maskrcnn_dentex.py --stage train
  python train_maskrcnn_dentex.py --stage prelabel --img-dir /kaggle/input/seu-dataset/images

Dependências (no Kaggle):
  detectron2, sam2, torch, opencv-python, tqdm, huggingface_hub
"""

import argparse
import os
import json
import random
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch

# ─────────────────────────────────────────────────────────────
# CONFIGURAÇÃO — ajuste antes de rodar
# ─────────────────────────────────────────────────────────────

DATASET_ROOT = "/kaggle/input/datasets/truthisneverlinear/dentex-challenge-2023"

PATHS = {
    "ann_enumeration": f"{DATASET_ROOT}/training_data/training_data/quadrant_enumeration/train_quadrant_enumeration.json",
    "xrays_enum":      f"{DATASET_ROOT}/training_data/training_data/quadrant_enumeration/xrays",
    "working":         "/kaggle/working",
    "prepared":        "/kaggle/working/dentex_maskrcnn",
    "checkpoints":     "/kaggle/working/output_maskrcnn",
    "pseudo_labels":   "/kaggle/working/pseudo_labels",
    "sam2_ckpt_dir":   "/kaggle/working/sam2_checkpoints",
}

# 32 classes = 4 quadrantes × 8 posições (notação FDI)
# category_id = quadrant_id_0based * 8 + tooth_pos_0based  →  0..31
NUM_CLASSES  = 32
TRAIN_ITERS  = 5000       # ~12 épocas com 507 imagens; aumente para fine-tune
BASE_LR      = 2.5e-4
SCORE_THRESH = 0.50       # confiança mínima para manter detecções

CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# Detectron2 redimensiona internamente para short_side=800, max_size=1333.
# Pré-redimensionar para esse limite reduz I/O no Kaggle sem perder informação
# (o compute da GPU seria o mesmo de qualquer jeito após o resize interno).
# As anotações usam coordenadas relativas salvas no JSON → precisam escalar junto.
PREP_MAX_SIZE = 1333


# ─────────────────────────────────────────────────────────────
# STAGE 1 — PREPARAÇÃO DO DATASET
# ─────────────────────────────────────────────────────────────

def apply_clahe_bgr(src: str, dst: str) -> tuple[float, float]:
    """
    Converte para grayscale, aplica CLAHE, redimensiona até PREP_MAX_SIZE no
    lado longo (mantendo aspect ratio) e salva como BGR 3-canais (R=G=B).

    Retorna (scale_x, scale_y) para que o chamador possa escalar anotações.
    Mantém consistência com 01_dados/preprocessar_panoramica.py (grayscale CLAHE).
    """
    img = cv2.imread(src, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Não foi possível abrir: {src}")

    h, w = img.shape
    scale = min(PREP_MAX_SIZE / max(h, w), 1.0)   # nunca amplia
    if scale < 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        scale = 1.0

    img_eq = CLAHE.apply(img)
    cv2.imwrite(dst, cv2.cvtColor(img_eq, cv2.COLOR_GRAY2BGR))
    return scale, scale


def prepare_dataset() -> None:
    """
    Converte as anotações DENTEX de enumeration para COCO padrão e
    aplica CLAHE em todas as imagens. Salva train/val splits 80/20.
    """
    img_dir = Path(PATHS["prepared"]) / "images"
    ann_dir = Path(PATHS["prepared"]) / "annotations"
    img_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)

    with open(PATHS["ann_enumeration"]) as f:
        raw = json.load(f)

    # Mapeia (quadrant_0based × 8 + tooth_0based) → categoria 0-31
    categories = [
        {"id": i, "name": f"tooth_{i + 1}", "supercategory": "tooth"}
        for i in range(NUM_CLASSES)
    ]

    # Anotações originais — convertemos categoria mas mantemos coords originais por ora.
    # O scale real é aplicado por imagem em build_split (depende do tamanho de cada imagem).
    ann_by_image: dict[int, list] = {}
    for ann in raw["annotations"]:
        cid = ann["category_id_1"] * 8 + ann["category_id_2"]
        entry = {
            "id":           ann["id"],
            "image_id":     ann["image_id"],
            "category_id":  cid,
            "bbox":         list(ann["bbox"]),
            "segmentation": [list(seg) for seg in ann["segmentation"]],
            "area":         ann["area"],
            "iscrowd":      ann.get("iscrowd", 0),
        }
        ann_by_image.setdefault(ann["image_id"], []).append(entry)

    all_ids = [img["id"] for img in raw["images"]]
    random.seed(42)
    random.shuffle(all_ids)
    split = int(0.8 * len(all_ids))
    train_ids = set(all_ids[:split])
    val_ids   = set(all_ids[split:])

    def _scale_ann(ann: dict, sx: float, sy: float) -> dict:
        """Escala bbox e polígonos de segmentação pelo fator (sx, sy)."""
        if sx == 1.0 and sy == 1.0:
            return ann
        x, y, w, h = ann["bbox"]
        ann["bbox"] = [x * sx, y * sy, w * sx, h * sy]
        ann["area"] = ann["area"] * sx * sy
        scaled_seg = []
        for poly in ann["segmentation"]:
            pts = []
            for i, v in enumerate(poly):
                pts.append(v * sx if i % 2 == 0 else v * sy)
            scaled_seg.append(pts)
        ann["segmentation"] = scaled_seg
        return ann

    def build_split(id_set: set, name: str) -> None:
        imgs, anns = [], []
        for img_info in tqdm(raw["images"], desc=f"CLAHE+resize {name}"):
            if img_info["id"] not in id_set:
                continue
            src = os.path.join(PATHS["xrays_enum"], img_info["file_name"])
            dst = str(img_dir / img_info["file_name"])
            sx, sy = apply_clahe_bgr(src, dst)

            # Atualiza width/height no manifesto se houve redimensionamento
            img_entry = dict(img_info)
            if sx != 1.0:
                img_entry["width"]  = int(img_info["width"]  * sx)
                img_entry["height"] = int(img_info["height"] * sy)
            imgs.append(img_entry)

            for ann in ann_by_image.get(img_info["id"], []):
                anns.append(_scale_ann(dict(ann), sx, sy))

        coco = {"images": imgs, "annotations": anns, "categories": categories}
        out = ann_dir / f"instances_{name}2017.json"
        with open(out, "w") as f:
            json.dump(coco, f)
        print(f"  {name}: {len(imgs)} imagens, {len(anns)} anotações → {out}")

    print("Preparando splits train/val...")
    build_split(train_ids, "train")
    build_split(val_ids,   "val")
    print("Dataset pronto.")


# ─────────────────────────────────────────────────────────────
# STAGE 2 — TREINO MASK R-CNN
# ─────────────────────────────────────────────────────────────

def train_maskrcnn() -> None:
    from detectron2 import model_zoo
    from detectron2.config import get_cfg
    from detectron2.data.datasets import register_coco_instances
    from detectron2.data import MetadataCatalog
    from detectron2.engine import DefaultTrainer

    img_dir = str(Path(PATHS["prepared"]) / "images")
    ann_dir = str(Path(PATHS["prepared"]) / "annotations")

    for split in ("train", "val"):
        name = f"dentex_{split}"
        if name not in MetadataCatalog:
            register_coco_instances(
                name, {},
                f"{ann_dir}/instances_{split}2017.json",
                img_dir,
            )

    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        )
    )
    cfg.DATASETS.TRAIN         = ("dentex_train",)
    cfg.DATASETS.TEST          = ("dentex_val",)
    cfg.DATALOADER.NUM_WORKERS = 2

    # Parte de pesos pré-treinados no COCO (transferência de aprendizado)
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    )

    cfg.SOLVER.IMS_PER_BATCH              = 2
    cfg.SOLVER.BASE_LR                    = BASE_LR
    cfg.SOLVER.MAX_ITER                   = TRAIN_ITERS
    cfg.SOLVER.STEPS                      = (int(TRAIN_ITERS * 0.70),
                                              int(TRAIN_ITERS * 0.90))
    cfg.SOLVER.CHECKPOINT_PERIOD          = 1000
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES          = NUM_CLASSES
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST    = SCORE_THRESH
    cfg.OUTPUT_DIR = PATHS["checkpoints"]
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    print(f"Iniciando treino: {TRAIN_ITERS} iterações, LR={BASE_LR}")
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
    print(f"Pesos salvos em: {cfg.OUTPUT_DIR}/model_final.pth")


# ─────────────────────────────────────────────────────────────
# STAGE 3 — PRE-LABEL (Mask R-CNN → SAM2)
# ─────────────────────────────────────────────────────────────

def load_maskrcnn_predictor():
    from detectron2 import model_zoo
    from detectron2.config import get_cfg
    from detectron2.data.datasets import register_coco_instances
    from detectron2.data import MetadataCatalog
    from detectron2.engine import DefaultPredictor

    img_dir = str(Path(PATHS["prepared"]) / "images")
    ann_dir = str(Path(PATHS["prepared"]) / "annotations")

    for split in ("train", "val"):
        name = f"dentex_{split}"
        if name not in MetadataCatalog:
            register_coco_instances(
                name, {},
                f"{ann_dir}/instances_{split}2017.json",
                img_dir,
            )

    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        )
    )
    cfg.DATASETS.TRAIN                       = ("dentex_train",)
    cfg.DATASETS.TEST                        = ("dentex_val",)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES          = NUM_CLASSES
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST    = SCORE_THRESH
    cfg.MODEL.WEIGHTS = os.path.join(PATHS["checkpoints"], "model_final.pth")
    cfg.DATALOADER.NUM_WORKERS = 0

    return DefaultPredictor(cfg)


def load_sam2_predictor():
    from huggingface_hub import hf_hub_download
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    ckpt = hf_hub_download(
        repo_id   = "facebook/sam2.1-hiera-base-plus",
        filename  = "sam2.1_hiera_base_plus.pt",
        local_dir = PATHS["sam2_ckpt_dir"],
    )
    print(f"SAM2 checkpoint: {ckpt}")

    model = build_sam2("sam2.1_hiera_b+.yaml", ckpt, device="cuda")
    predictor = SAM2ImagePredictor(model)
    print("SAM2 pronto.")
    return predictor


def prelabel_images(img_dir: str) -> None:
    """
    Para cada imagem em img_dir:
      1. Mask R-CNN detecta dentes (bounding box + máscara inicial)
      2. SAM2 refina cada box em máscara de alta qualidade
      3. Salva JSON com anotações por imagem em PATHS["pseudo_labels"]

    JSON de saída por imagem:
      { file_name, height, width,
        annotations: [{category_id, bbox [xywh], segmentation [[pts]], score, sam_score}] }
    """
    out_dir = Path(PATHS["pseudo_labels"])
    out_dir.mkdir(parents=True, exist_ok=True)

    img_files = sorted(
        p for p in Path(img_dir).iterdir()
        if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    print(f"Imagens encontradas: {len(img_files)}")

    if not img_files:
        raise FileNotFoundError(f"Nenhuma imagem em: {img_dir}")

    print("Carregando Mask R-CNN...")
    mrcnn = load_maskrcnn_predictor()

    print("Carregando SAM2...")
    sam2  = load_sam2_predictor()

    skipped = 0
    for img_path in tqdm(img_files, desc="Pre-labeling"):
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            skipped += 1
            continue

        # As imagens privadas já saem de 01_dados/preprocessar_panoramica.py como grayscale.
        # Converte para grayscale (no-op se já for), aplica CLAHE, expande para
        # 3 canais — exatamente igual ao que apply_clahe_bgr faz no treino.
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim == 3 else img_bgr
        img_eq = cv2.cvtColor(CLAHE.apply(gray), cv2.COLOR_GRAY2BGR)

        # ── Mask R-CNN ───────────────────────────────────────────────
        outputs   = mrcnn(img_eq)
        instances = outputs["instances"].to("cpu")
        scores    = instances.scores.numpy()
        keep      = scores >= SCORE_THRESH

        if keep.sum() == 0:
            skipped += 1
            continue

        boxes   = instances.pred_boxes.tensor.numpy()[keep]   # (N,4) xyxy
        classes = instances.pred_classes.numpy()[keep]
        scores  = scores[keep]

        # ── SAM2: refina cada box em máscara precisa ─────────────────
        img_rgb = cv2.cvtColor(img_eq, cv2.COLOR_BGR2RGB)
        sam2.set_image(img_rgb)

        annotations = []
        for i in range(len(boxes)):
            masks, sam_scores, _ = sam2.predict(
                box              = boxes[i][None],   # (1, 4) xyxy
                multimask_output = False,
            )
            mask = masks[0, 0].astype(np.uint8)

            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            seg = [c.flatten().tolist() for c in contours if len(c) >= 3]
            if not seg:
                continue

            x1, y1, x2, y2 = boxes[i]
            annotations.append({
                "category_id": int(classes[i]),
                "bbox":        [float(x1), float(y1),
                                float(x2 - x1), float(y2 - y1)],
                "segmentation": seg,
                "score":       float(scores[i]),
                "sam_score":   float(sam_scores[0]),
            })

        result = {
            "file_name":   img_path.name,
            "height":      img_rgb.shape[0],
            "width":       img_rgb.shape[1],
            "annotations": annotations,
        }
        out_file = out_dir / (img_path.stem + ".json")
        with open(out_file, "w") as f:
            json.dump(result, f)

    total = len(img_files)
    print(f"\nConcluído: {total - skipped}/{total} imagens anotadas")
    print(f"JSONs salvos em: {out_dir}")
    if skipped:
        print(f"  {skipped} imagens puladas (sem detecção ou falha de leitura)")


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Treina Mask R-CNN no DENTEX e gera pseudo-labels com SAM2."
    )
    parser.add_argument(
        "--stage", required=True,
        choices=["prep", "train", "prelabel", "all"],
        help="Estágio a executar. 'all' executa prep→train→prelabel em sequência.",
    )
    parser.add_argument(
        "--img-dir",
        default="/kaggle/input/seu-dataset/images",
        help="Diretório com as imagens do dataset privado (somente para --stage prelabel).",
    )
    args = parser.parse_args()

    run_all     = args.stage == "all"
    run_prep    = run_all or args.stage == "prep"
    run_train   = run_all or args.stage == "train"
    run_prelabel = run_all or args.stage == "prelabel"

    if run_prep:
        print("\n" + "=" * 60)
        print("ESTÁGIO 1: Preparando dataset DENTEX")
        print("=" * 60)
        prepare_dataset()

    if run_train:
        print("\n" + "=" * 60)
        print("ESTÁGIO 2: Treinando Mask R-CNN")
        print("=" * 60)
        train_maskrcnn()

    if run_prelabel:
        print("\n" + "=" * 60)
        print("ESTÁGIO 3: Gerando pseudo-labels")
        print("=" * 60)
        prelabel_images(args.img_dir)


if __name__ == "__main__":
    main()
