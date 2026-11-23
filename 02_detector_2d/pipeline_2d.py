"""Detector 2D na panorâmica (Mask R-CNN, 32 classes FDI) e extração dos tokens por dente.

Pipeline de inferência (o mesmo da célula "Funções do pipeline NOVO" do tcc-xray-v2.ipynb, que roda no Kaggle):
recorte das barras pretas + ampliação para 1333 px de largura → Mask R-CNN → score >= 0,5 → NMS agnóstico →
1 detecção por número + resgate (com ou sem a correção de lado pela linha média).

Usado por:
  montar_cvat_teste_multimodal.py   pré-rótulos do conjunto de teste para anotar no CVAT (predict_novo + to_annotations)
  extrair_features_2d.py            tokens 2D de cada caso (instalar_captura + extrair)
  01_dados/escolher_panoramica.py   escolha da panorâmica entre as candidatas de um caso
  07_inferencia/rodar.py            inferência ponta a ponta
"""
from pathlib import Path

import cv2
import numpy as np
import torch
from pycocotools import mask as mask_util
from torchvision.ops import nms
import PIL.Image

# detectron2 0.6 usa nomes do Pillow que saíram nas versões novas (mesmo remendo do montar_cvat_teste_multimodal.py)
for antigo, novo in [("LINEAR", "BILINEAR"), ("CUBIC", "BICUBIC"), ("ANTIALIAS", "LANCZOS")]:
    if not hasattr(PIL.Image, antigo):
        setattr(PIL.Image, antigo, getattr(PIL.Image, novo))

AQUI = Path(__file__).resolve().parent
CFG = AQUI / "cfg" / "COCO-InstanceSegmentation" / "mask_rcnn_R_50_FPN_3x.yaml"
NUM_CLASSES, SCORE_THRESH_TEST, SCORE_MIN, LARGURA_AMPLIADA = 32, 0.05, 0.50, 1333


def imread(p):
    img = cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"não consegui ler a imagem {p} (use PNG ou JPG)")
    return img


def install_prob_hook(predictor):
    """Guarda as probabilidades das 32 classes de cada detecção (usadas no resgate)."""
    bp = predictor.model.roi_heads.box_predictor
    orig = bp.inference

    def inference(predictions, proposals):
        probs = bp.predict_probs(predictions, proposals)
        instances, kept = orig(predictions, proposals)
        for inst, pr, k in zip(instances, probs, kept):
            inst.probs = pr[k][:, :-1]
        return instances, kept
    bp.inference = inference


def build_predictor(weights, device=None):
    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor
    cfg = get_cfg()
    cfg.merge_from_file(str(CFG))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = NUM_CLASSES
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = SCORE_THRESH_TEST
    cfg.MODEL.WEIGHTS = str(weights)
    cfg.MODEL.DEVICE = device or ("cuda" if torch.cuda.is_available() else "cpu")
    pred = DefaultPredictor(cfg)
    install_prob_hook(pred)
    return pred


def crop_upscale(img, target_w=LARGURA_AMPLIADA):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rows = np.where(g.max(axis=1) > 10)[0]
    cols = np.where(g.max(axis=0) > 10)[0]
    if len(rows) == 0 or len(cols) == 0:
        rows, cols = np.arange(img.shape[0]), np.arange(img.shape[1])
    y0, y1, x0, x1 = rows[0], rows[-1] + 1, cols[0], cols[-1] + 1
    c = img[y0:y1, x0:x1]
    s = target_w / c.shape[1]
    up = cv2.resize(c, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
    return up, (s, int(x0), int(y0), int(x1 - x0), int(y1 - y0))


def to_input_space(p, s, x0, y0, H, W, score_min):
    sel = p["scores"] >= score_min
    p = {k: v[sel] for k, v in p.items()}
    masks = []
    for m in p["masks"]:
        m = m.astype(np.uint8)
        h, w = m.shape
        small = cv2.resize(m, (int(round(w / s)), int(round(h / s))), interpolation=cv2.INTER_NEAREST)
        full = np.zeros((H, W), np.uint8)
        hh, ww = min(small.shape[0], H - y0), min(small.shape[1], W - x0)
        full[y0:y0 + hh, x0:x0 + ww] = small[:hh, :ww]
        masks.append(full)
    p["masks"] = np.array(masks) if masks else np.zeros((0, H, W), np.uint8)
    b = p["boxes"] / s + np.array([x0, y0, x0, y0], np.float32)
    p["boxes"] = np.clip(b, 0, [W, H, W, H]).astype(np.float32)
    return p


def postprocess(p, iou=0.5):
    if len(p["scores"]) == 0:
        return p
    keep = nms(torch.tensor(p["boxes"]), torch.tensor(p["scores"]), iou).numpy()
    best = {}
    for i in keep:
        c = p["classes"][i]
        if c not in best or p["scores"][i] > p["scores"][best[c]]:
            best[c] = i
    idx = np.array(sorted(best.values(), key=lambda i: -p["scores"][i]))
    return {k: v[idx] for k, v in p.items()}


def fix_side(classes, cx, mid_x):
    upper = np.isin((classes - 1) // 8, [0, 1])
    pos = (classes - 1) % 8
    left = cx < mid_x
    q = np.where(upper, np.where(left, 0, 1), np.where(left, 3, 2))
    return q * 8 + pos + 1


def number_with_rescue(p, mid_x, usar_linha_media=True, overlap_max=0.3):
    if len(p["scores"]) == 0:
        return p
    cx = np.array([np.nonzero(m)[1].mean() if m.any() else (b[0] + b[2]) / 2
                   for m, b in zip(p["masks"], p["boxes"])])
    cls0 = fix_side(p["classes"], cx, mid_x) if usar_linha_media else p["classes"].copy()
    q = dict(p, classes=cls0, idx=np.arange(len(cx)))
    kept = postprocess(q)
    kset = set(kept["idx"].tolist())
    cls = {int(i): int(c) for i, c in zip(kept["idx"], kept["classes"])}
    km = {i: p["masks"][i].astype(bool) for i in kset}
    pr = p["probs"].reshape(-1, 4, 8)
    cand = sorted((i for i in range(len(cx)) if i not in kset), key=lambda i: -p["scores"][i])
    for i in cand:
        m = p["masks"][i].astype(bool)
        if any((m & km[j]).sum() / max((m | km[j]).sum(), 1) > overlap_max for j in km):
            continue
        quad = (int(cls0[i]) - 1) // 8
        same = [j for j in cls if (cls[j] - 1) // 8 == quad]
        d = abs(cx[i] - mid_x)
        inner = [(cls[j] - 1) % 8 for j in same if abs(cx[j] - mid_x) < d]
        outer = [(cls[j] - 1) % 8 for j in same if abs(cx[j] - mid_x) > d]
        lo = max(inner) + 1 if inner else 0
        hi = min(outer) - 1 if outer else 7
        used = {(cls[j] - 1) % 8 for j in same}
        free = [ps for ps in range(lo, hi + 1) if ps not in used]
        if not free:
            continue
        arch = [0, 1] if quad in (0, 1) else [2, 3]
        pp = pr[i, arch].sum(0)
        ps = max(free, key=lambda k: pp[k])
        cls[i] = quad * 8 + ps + 1
        km[i] = m
    order = sorted(cls)
    out = {k: v[order] for k, v in p.items()}
    out["classes"] = np.array([cls[i] for i in order])
    return out


# ── Pré-rótulo (montar_cvat_teste_multimodal.py): as mesmas funções da célula do notebook ──────────────────
def predict(pred, img):
    """Saída crua do modelo. classes em 1..32."""
    inst = pred(img)["instances"].to("cpu")
    return dict(boxes=inst.pred_boxes.tensor.numpy(), scores=inst.scores.numpy(),
                classes=inst.pred_classes.numpy() + 1, masks=inst.pred_masks.numpy(),
                probs=inst.probs.numpy())


def letterbox512(img):
    """Igual ao letterbox de 01_dados/preprocessar_panoramica.py. Retorna a imagem e (escala, pad_x, pad_y)."""
    h, w = img.shape[:2]
    s = min(512 / w, 512 / h)
    nw, nh = int(w * s), int(h * s)
    r = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    out = np.zeros((512, 512, 3), np.uint8)
    px, py = (512 - nw) // 2, (512 - nh) // 2
    out[py:py + nh, px:px + nw] = r
    return out, (s, px, py)


def novo_bruto(pred, img):
    """Recorte + ampliação + inferência, com tudo devolvido no espaço da imagem de entrada.
    Retorna (predições com score >= SCORE_MIN, x da linha média)."""
    H, W = img.shape[:2]
    up, (s, x0, y0, cw, ch) = crop_upscale(img)
    p = to_input_space(predict(pred, up), s, x0, y0, H, W, SCORE_MIN)
    return p, x0 + cw / 2


def predict_novo(pred, img, usar_linha_media=True):
    """Pipeline NOVO completo."""
    p, mid_x = novo_bruto(pred, img)
    return number_with_rescue(p, mid_x, usar_linha_media)


def to_annotations(p):
    """Predições -> anotações no formato dos JSONs de pré-rótulo (category_id 1..32, bbox xywh)."""
    anns = []
    for b, s, c, m in zip(p["boxes"], p["scores"], p["classes"], p["masks"]):
        contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        seg = [ct.flatten().tolist() for ct in contours if len(ct) >= 3]
        if not seg:
            continue
        x1, y1, x2, y2 = [float(v) for v in b]
        anns.append({"category_id": int(c), "bbox": [x1, y1, x2 - x1, y2 - y1],
                     "segmentation": seg, "score": float(s)})
    return anns


# ── Tokens 2D (extrair_features_2d.py, 07_inferencia/rodar.py) ─────────────────────────────────────────────
def instalar_captura(pred):
    m, cap = pred.model, {}
    m.backbone.register_forward_hook(
        lambda mod, i, o: cap.__setitem__(
            "fpn", np.concatenate([o[k].mean((2, 3))[0].detach().cpu().numpy() for k in sorted(o)])))
    m.roi_heads.box_head.register_forward_hook(lambda mod, i, o: cap.__setitem__("box", o.detach().cpu()))
    m.roi_heads.mask_head.predictor.register_forward_hook(
        lambda mod, i, o: cap.update(mask_in=i[0].detach().cpu(), mask_logit=o.detach().cpu()))
    bp = m.roi_heads.box_predictor
    orig = bp.inference

    def inference(predictions, proposals):
        instances, kept = orig(predictions, proposals)
        cap["kept"] = kept[0].detach().cpu()
        return instances, kept
    bp.inference = inference
    return cap


def predict_com_features(pred, cap, img):
    cap.clear()
    inst = pred(img)["instances"].to("cpu")
    n = len(inst)
    cls = inst.pred_classes
    if n:
        assert len(cap["kept"]) == n and cap["mask_in"].shape[0] == n, "hooks desalinhados"
        feat_box = cap["box"][cap["kept"]].numpy()
        feat_mask = cap["mask_in"].mean((2, 3)).numpy()
        mask28 = torch.sigmoid(cap["mask_logit"][torch.arange(n), cls]).numpy()
    else:
        feat_box, feat_mask, mask28 = np.zeros((0, 1024)), np.zeros((0, 256)), np.zeros((0, 28, 28))
    return dict(boxes=inst.pred_boxes.tensor.numpy(), scores=inst.scores.numpy(),
                classes=cls.numpy() + 1, classe_modelo=cls.numpy() + 1, masks=inst.pred_masks.numpy(),
                probs=inst.probs.detach().cpu().numpy(), feat_box=feat_box, feat_mask=feat_mask,
                mask28=mask28), cap["fpn"]


def extrair(pred, cap, img):
    """Dicionário com os mesmos campos dos features_2d_flipswap/<case_id>.npz do TCC."""
    H, W = img.shape[:2]
    up, (s, x0, y0, cw, ch) = crop_upscale(img)
    bruto, fpn = predict_com_features(pred, cap, up)
    p = to_input_space(bruto, s, x0, y0, H, W, SCORE_MIN)
    p = number_with_rescue(p, x0 + cw / 2, usar_linha_media=False)
    n = len(p["scores"])
    cen = np.array([[np.nonzero(m)[1].mean(), np.nonzero(m)[0].mean()] if m.any()
                    else [(b[0] + b[2]) / 2, (b[1] + b[3]) / 2] for m, b in zip(p["masks"], p["boxes"])]).reshape(n, 2)
    fdi = np.array([(c - 1) // 8 * 10 + (c - 1) % 8 + 11 for c in p["classes"]], np.int16)
    rle = [mask_util.encode(np.asfortranarray(m.astype(np.uint8)))["counts"].decode() for m in p["masks"]]
    return dict(
        fdi=fdi, classe=np.asarray(p["classes"]).astype(np.int16),
        classe_modelo=np.asarray(p["classe_modelo"]).astype(np.int16),
        arcada=np.where(fdi < 30, 0, 1).astype(np.int8),
        score=p["scores"].astype(np.float32), probs=p["probs"].astype(np.float32),
        caixa=p["boxes"].astype(np.float32), centroide=cen.astype(np.float32),
        pos_norm=((cen - [x0, y0]) / [cw, ch]).astype(np.float32),
        area=p["masks"].sum((1, 2)).astype(np.int32),
        feat_box=p["feat_box"].astype(np.float16), feat_mask=p["feat_mask"].astype(np.float16),
        mask28=p["mask28"].astype(np.float16), mask_rle=np.array(rle, dtype=str),
        fpn_global=fpn.astype(np.float16),
        recorte=np.array([s, x0, y0, cw, ch], np.float32), tamanho=np.array([H, W], np.int32)), p["masks"]
