"""Pré-processamento das radiografias panorâmicas (o mesmo aplicado a toda a coorte do TCC).

Para cada imagem bruta: recorte automático da região da radiografia (smart_crop_xray), filtro heurístico de
panorâmica (is_panoramic_xray), corte das margens (tarjas com dados do paciente), CLAHE e letterbox para 512×512.

Uso:
  python 01_dados/preprocessar_panoramica.py --entrada <pasta com <case_id>.<ext>> --saida <pasta>
Saídas:
  <saida>/processados/<case_id>.png              panorâmica válida, pré-processada (512×512, uint8)
  <saida>/rejeitados/<case_id>__<motivo>.png     imagem bruta reprovada pelo filtro (para revisão manual)
  <saida>/auditoria.csv                          case_id, status, motivo
Depois da revisão manual, as aprovadas vão para <TCC_DADOS>/panoramicas/<lote>/processados/.
"""
import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

# Tamanho final de saída
TARGET_SIZE: tuple[int, int] = (512, 512)

# Margens de corte em pixels — remove tarjas/textos nas bordas
CROP_TOP = 20
CROP_BOTTOM = 20
CROP_LEFT = 10
CROP_RIGHT = 10

# CLAHE — padronização de contraste
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID = (8, 8)

# ─────────────────────────────────────────────
# PIPELINE DE PRÉ-PROCESSAMENTO (opera na RAM)
# ─────────────────────────────────────────────

def crop_margins(img: np.ndarray) -> np.ndarray:
    """Remove bordas para eliminar tarjas/textos com dados do paciente (LGPD)."""
    h, w = img.shape
    t, b, l, r = CROP_TOP, h - CROP_BOTTOM, CROP_LEFT, w - CROP_RIGHT
    if b <= t or r <= l:
        raise ValueError(
            f"Margens ({CROP_TOP},{CROP_BOTTOM},{CROP_LEFT},{CROP_RIGHT}) "
            f"excedem dimensões da imagem ({w}×{h})."
        )
    return img[t:b, l:r]


def _trim_to_panoramic(img: np.ndarray) -> np.ndarray:
    """
    Remove linhas com baixa densidade de tecido radiográfico.

    A panorâmica tem tecido ósseo/dentário distribuído em toda a largura →
    alta fração de pixels em [25, 220] por linha. Cabeçalhos (texto + logo),
    thumbnails periapicais/cefalométricos e rótulos laterais têm fração muito
    menor, criando vales no perfil que separam os segmentos.

    Mantém apenas o MAIOR segmento contínuo acima de 35% do pico.
    Para imagens limpas (sem extras), um único segmento cobre tudo → sem corte.
    """
    h, w = img.shape
    if h < 40 or w < 40:
        return img

    tissue = ((img >= 25) & (img <= 220)).astype(np.float32)

    # Janela pequena (h//40) para preservar gaps curtos entre panorâmica e thumbnails
    win = max(h // 40, 2)
    row_prof = np.convolve(tissue.mean(axis=1), np.ones(win) / win, mode='same')

    peak = row_prof.max()
    if peak < 0.08:
        return img

    thr = peak * 0.35
    above = row_prof >= thr

    segs: list[tuple[int, int]] = []
    seg_start: int | None = None
    for i, v in enumerate(above):
        if v and seg_start is None:
            seg_start = i
        elif not v and seg_start is not None:
            segs.append((seg_start, i - 1))
            seg_start = None
    if seg_start is not None:
        segs.append((seg_start, len(above) - 1))

    if not segs:
        return img

    r0, r1 = max(segs, key=lambda s: s[1] - s[0])

    pad = 8
    return img[max(0, r0 - pad): min(h, r1 + 1 + pad), :]


def smart_crop_xray(img: np.ndarray) -> np.ndarray:
    """
    Extrai automaticamente a região da radiografia removendo cabeçalhos,
    rodapés, laudos e formulários.

    Algoritmo:
      1. Blur agressivo (21×21) dilui texto fino até abaixo do limiar
      2. Máscara [20, 240] captura tecido do RX, exclui fundo preto e papel branco
      3. Fechamento morfológico une fragmentos do RX em um blob contínuo
      4. Seleção por score (área × 3 + bônus landscape + bônus escuro)
      5. _trim_to_panoramic: perfil de densidade de tecido remove cabeçalhos,
         thumbnails periapicais/cefalométricos e rótulos laterais

    Para uma imagem já limpa, ambas as etapas retornam essencialmente o original.
    """
    h, w = img.shape

    blurred = cv2.GaussianBlur(img, (21, 21), 0)

    mask = np.zeros_like(blurred, dtype=np.uint8)
    mask[(blurred >= 20) & (blurred <= 240)] = 255

    ksize = min(max(w, h) // 30, 41) | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img

    min_area = h * w * 0.06
    big = [c for c in contours if cv2.contourArea(c) >= min_area]
    if not big:
        return img

    def _blob_score(c: np.ndarray) -> float:
        bx, by, bw, bh = cv2.boundingRect(c)
        ar = bw / max(bh, 1)
        region_mean = float(img[by:by + bh, bx:bx + bw].mean())
        area_frac   = cv2.contourArea(c) / (h * w)
        ar_score    = min(ar / 1.5, 2.0)
        dark_score  = max(0.0, 1.0 - region_mean / 180.0)
        return area_frac * 3.0 + ar_score + dark_score

    best = max(big, key=_blob_score)
    bx, by, bw, bh = cv2.boundingRect(best)
    pad = 10
    coarse = img[max(0, by - pad): min(h, by + bh + pad),
                 max(0, bx - pad): min(w, bx + bw + pad)]

    return _trim_to_panoramic(coarse)


def apply_clahe(img: np.ndarray) -> np.ndarray:
    """Equalização adaptativa de histograma: padroniza contraste entre imagens escuras e claras."""
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID)
    return clahe.apply(img)


def letterbox(img: np.ndarray, target: tuple[int, int] = TARGET_SIZE) -> np.ndarray:
    """
    Redimensiona preservando aspect ratio e centraliza em canvas preto.
    Evita distorção da anatomia da mandíbula que um resize direto causaria.
    """
    h_src, w_src = img.shape
    h_tgt, w_tgt = target
    scale    = min(w_tgt / w_src, h_tgt / h_src)
    new_w    = int(w_src * scale)
    new_h    = int(h_src * scale)
    resized  = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas   = np.zeros((h_tgt, w_tgt), dtype=np.uint8)
    pad_top  = (h_tgt - new_h) // 2
    pad_left = (w_tgt - new_w) // 2
    canvas[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized
    return canvas


def normalize(img: np.ndarray) -> np.ndarray:
    """uint8 [0,255] → float32 [0.0, 1.0] (aplicado no data loader em produção)."""
    return img.astype(np.float32) / 255.0


def preprocess(img: np.ndarray) -> np.ndarray:
    """Pipeline completo em memória: smart_crop → crop_margins → CLAHE → letterbox → normalize."""
    img = smart_crop_xray(img)
    img = crop_margins(img)
    img = apply_clahe(img)
    img = letterbox(img)
    return normalize(img)


# ─────────────────────────────────────────────
# FILTROS DE ARQUIVO E VALIDAÇÃO DE RAIO X
# ─────────────────────────────────────────────

# Extensões que o cv2 nunca consegue decodificar como imagem
_NON_IMAGE_EXT = {".pdf", ".doc", ".docx", ".xls", ".xlsx", ".txt", ".rtf"}

# Thresholds do filtro heurístico (aplicados APÓS smart_crop)
_MIN_ASPECT   = 1.0    # panorâmicas + layouts com thumbnails/header chegam a ~1.1
_MAX_WHITE    = 0.35   # fração de pixels > 230 — documento escaneado ainda presente
_MIN_STD      = 20.0   # desvio-padrão mínimo — imagem homogênea demais → não é RX
_MAX_MEAN     = 185.0  # muito claro → foto ou documento
_MIN_MEAN     = 20.0   # quase preto → corrompido


def _reject_by_path(path: str) -> str | None:
    """Rejeita antes do download por extensão ou nome de arquivo inválido."""
    fname = path.rsplit("/", 1)[-1]
    if fname.startswith("._"):
        return "macOS resource fork (._)"
    ext = ("." + fname.rsplit(".", 1)[-1].lower()) if "." in fname else ""
    if ext in _NON_IMAGE_EXT:
        return f"extensão não-imagem ({ext})"
    return None


def is_panoramic_xray(img: np.ndarray) -> tuple[bool, str]:
    """
    Verifica heuristicamente se a imagem (pós smart_crop) é uma panorâmica válida.
    Retorna (True, "ok") ou (False, motivo).

    Critérios:
      - aspect ≥ 1.0  : exclui retratos/quadrados (foto de rosto, laudo A4 não cortado)
      - white ≤ 35%   : exclui documentos escaneados com muito papel branco
      - std ≥ 20      : exclui imagens planas/sem contraste (substitui dark_ratio)
      - mean [20,185] : exclui imagens quase-pretas ou super-claras
    """
    h, w   = img.shape
    aspect = w / h
    mean   = float(img.mean())
    std    = float(img.std())
    white  = float((img > 230).mean())

    if aspect < _MIN_ASPECT:
        return False, f"aspect={aspect:.2f} — retrato/quadrado (laudo, foto, cefalometria)"
    if white > _MAX_WHITE:
        return False, f"white_ratio={white:.1%} — documento escaneado ainda presente"
    if std < _MIN_STD:
        return False, f"std={std:.1f} — imagem homogênea demais"
    if mean > _MAX_MEAN:
        return False, f"mean={mean:.1f} — imagem muito clara, provável foto/doc"
    if mean < _MIN_MEAN:
        return False, f"mean={mean:.1f} — imagem quase preta, possível falha"
    return True, "ok"


EXTENSOES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def processar_arquivo(arq: Path, dir_ok: Path, dir_rej: Path) -> tuple[str, str]:
    """Pré-processa uma imagem; devolve (status, motivo)."""
    case_id = arq.stem.split(".")[0]
    img = cv2.imdecode(np.frombuffer(arq.read_bytes(), np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return "erro", "cv2.imdecode falhou"
    recorte = smart_crop_xray(img)
    ok, motivo = is_panoramic_xray(recorte)
    if not ok:
        slug = motivo.split("—")[0].strip()[:40].replace(" ", "_").replace("%", "pct")
        cv2.imwrite(str(dir_rej / f"{case_id}__{slug}.png"), img)
        return "rejeitado", motivo
    final = letterbox(apply_clahe(crop_margins(recorte)))
    cv2.imwrite(str(dir_ok / f"{case_id}.png"), final)
    return "ok", ""


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--entrada", type=Path, required=True, help="pasta com as panorâmicas brutas (<case_id>.<ext>)")
    ap.add_argument("--saida", type=Path, required=True)
    a = ap.parse_args()
    dir_ok, dir_rej = a.saida / "processados", a.saida / "rejeitados"
    dir_ok.mkdir(parents=True, exist_ok=True)
    dir_rej.mkdir(parents=True, exist_ok=True)
    linhas = []
    for arq in sorted(p for p in a.entrada.iterdir() if p.suffix.lower() in EXTENSOES and not p.name.startswith("._")):
        try:
            status, motivo = processar_arquivo(arq, dir_ok, dir_rej)
        except Exception as ex:                 # uma imagem ruim não para a rodada
            status, motivo = "erro", str(ex)
        linhas.append(dict(case_id=arq.stem.split(".")[0], status=status, motivo=motivo))
        print(f"{arq.name}: {status} {motivo}", flush=True)
    with open(a.saida / "auditoria.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["case_id", "status", "motivo"])
        w.writeheader()
        w.writerows(linhas)
    n_ok = sum(l["status"] == "ok" for l in linhas)
    print(f"{n_ok}/{len(linhas)} aprovadas → {dir_ok}")


if __name__ == "__main__":
    main()
