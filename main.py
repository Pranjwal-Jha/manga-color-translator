#!/usr/bin/env python3
"""
Manga OCR pipeline
==================
Stage 1 — Detection : EasyOCR's CRAFT-based detector finds individual text
                      regions reliably instead of coarse whole-panel strips.
Stage 2 — Recognition: manga-ocr, specifically trained on manga glyphs,
                      handles each crop.  CRAFT + manga-ocr outperforms
                      either tool running alone on manga pages.
Stage 3 — Reading order: right-to-left, top-to-bottom (standard manga).

Usage
-----
  uv run python main.py --image manga_translate_text.png
  uv run python main.py --image page.jpg --debug --debug-image out.png --save-crops crops/
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Tuple

import cv2
import easyocr
import numpy as np
from manga_ocr import MangaOcr
from PIL import Image

# ─────────────────────────────────────────────────────────────────────────────
# Types
# ─────────────────────────────────────────────────────────────────────────────

# Axis-aligned bounding box: (x1, y1, x2, y2)
Box = Tuple[int, int, int, int]


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 – CRAFT detection (via EasyOCR)
# ─────────────────────────────────────────────────────────────────────────────

def detect_text_regions(img_bgr: np.ndarray, gpu: bool = True) -> List[Box]:
    """
    Run EasyOCR's CRAFT detector and return axis-aligned bounding boxes.

    We call reader.detect() (detection only) so that recognition is handled
    entirely by manga-ocr, which is far better tuned for manga glyphs.

    EasyOCR detect() returns:
      horizontal_list[0] – list of [x_min, x_max, y_min, y_max]
      free_list[0]       – list of 4-point quads for rotated/vertical text
    """
    reader = easyocr.Reader(
        ['ja'],
        gpu=gpu,
        verbose=False,
    )

    img_h, img_w = img_bgr.shape[:2]

    horizontal_list, free_list = reader.detect(
        img_bgr,
        min_size=10,
        text_threshold=0.55,     # lower = more sensitive detection
        low_text=0.30,
        link_threshold=0.25,     # lower = links characters into blocks better
        canvas_size=2560,
        mag_ratio=1.5,           # slight up-scale helps small/dense manga text
        slope_ths=0.4,           # tolerate mild tilt
        ycenter_ths=0.5,
        height_ths=0.5,
        width_ths=0.5,
        add_margin=0.12,         # pad boxes slightly so characters aren't clipped
    )

    boxes: List[Box] = []

    for entry in horizontal_list[0]:
        x1, x2, y1, y2 = (int(v) for v in entry)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img_w, x2), min(img_h, y2)
        if x2 - x1 >= 8 and y2 - y1 >= 8:
            boxes.append((x1, y1, x2, y2))

    # free_list catches vertical text columns that CRAFT found as quads
    for quad in free_list[0]:
        pts = np.array(quad, dtype=np.int32)
        rx, ry, rw, rh = cv2.boundingRect(pts)
        x1 = max(0, rx)
        y1 = max(0, ry)
        x2 = min(img_w, rx + rw)
        y2 = min(img_h, ry + rh)
        if x2 - x1 >= 8 and y2 - y1 >= 8:
            boxes.append((x1, y1, x2, y2))

    return boxes


# ─────────────────────────────────────────────────────────────────────────────
# Box post-processing (filter, NMS, sort)
# ─────────────────────────────────────────────────────────────────────────────

def _iou(a: Box, b: Box) -> float:
    ix1 = max(a[0], b[0]);  iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]);  iy2 = min(a[3], b[3])
    iw = max(0, ix2 - ix1); ih = max(0, iy2 - iy1)
    inter = iw * ih
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def nms(boxes: List[Box], threshold: float = 0.4) -> List[Box]:
    """Non-maximum suppression — keeps larger box when two overlap."""
    if not boxes:
        return []
    by_area = sorted(boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True)
    keep: List[Box] = []
    while by_area:
        cur = by_area.pop(0)
        keep.append(cur)
        by_area = [b for b in by_area if _iou(cur, b) < threshold]
    return keep


def filter_boxes(boxes: List[Box], img_w: int, img_h: int) -> List[Box]:
    """
    Remove boxes that are clearly not text:
      - Too small (noise blobs)
      - Too large (covers most of the panel — hallucination magnet)
      - Absurd aspect ratio unlikely for any real text block
    """
    img_area = img_w * img_h
    out: List[Box] = []
    for x1, y1, x2, y2 in boxes:
        bw, bh = x2 - x1, y2 - y1
        area = bw * bh
        if area < 80:
            continue
        if area > img_area * 0.40:        # skip anything covering >40% of image
            continue
        aspect = bw / max(bh, 1)
        if aspect > 30 or aspect < 0.03:  # allow both wide and tall text columns
            continue
        out.append((x1, y1, x2, y2))
    return out


def sort_manga_order(boxes: List[Box], img_h: int) -> List[Box]:
    """
    Approximate manga reading order: right-to-left, top-to-bottom.
    Boxes are bucketed into horizontal bands (≈1/14 of image height) so that
    boxes at roughly the same height sort by descending x-centre.
    """
    band = max(30, img_h // 14)
    return sorted(
        boxes,
        key=lambda b: (
            (b[1] + b[3]) // 2 // band,   # vertical band
            -((b[0] + b[2]) // 2),         # right-to-left within band
        ),
    )


def filter_furigana(boxes: List[Box], img_h: int) -> List[Box]:
    """
    Drop boxes that are almost certainly furigana (ruby annotations) or
    noise blobs too small to hold a readable character.

    Furigana sit above/beside their parent kanji and are typically ⅓–¼ the
    height of the main-text characters.  Any detection shorter than ~30 px on
    a standard manga scan is not worth passing to manga-ocr — it would only
    produce hallucinations.

    Thresholds scale with image height so the filter adapts to resolution:
      min_h   ≈ 1.6 % of image height  →  ~30 px on a 1884-px-tall scan
      min_w   ≈ half of min_h          →  allows narrow single-character columns
      min_area = min_h²                →  rejects tiny square blobs
    """
    min_h    = max(28, int(img_h * 0.016))
    min_w    = max(14, min_h // 2)
    min_area = min_h * min_h
    return [
        (x1, y1, x2, y2) for x1, y1, x2, y2 in boxes
        if (y2 - y1) >= min_h
        and (x2 - x1) >= min_w
        and (y2 - y1) * (x2 - x1) >= min_area
    ]


def merge_nearby_boxes(boxes: List[Box], gap_ratio: float = 0.25) -> List[Box]:
    """
    Collapse spatially close boxes into one, handling two common CRAFT
    fragmentation patterns:

      1. Vertical text columns — CRAFT often detects each column of a speech
         bubble as a separate narrow box.  These sit side-by-side with nearly
         identical vertical extents and should become one crop.

      2. Minor fragmentation — a single text run whose left/right edge was
         slightly under the detection threshold gets snapped into two boxes.

    Algorithm
    ---------
    Each box is padded by  gap_ratio × its own height  on every side before
    the overlap test, so the merge threshold naturally scales with text size
    (small boxes need a shorter absolute gap, large boxes tolerate a bigger
    one).  The cap of 60 px prevents very tall speech-bubble boxes from
    accidentally pulling in text from the next panel.

    Union-Find is used so transitive chains (A touches B, B touches C →
    all three merge) are resolved in a single O(n²) pass instead of requiring
    multiple iterations.
    """
    if len(boxes) <= 1:
        return boxes

    from collections import defaultdict

    n = len(boxes)

    def padded_overlap(a: Box, b: Box) -> bool:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        # Scale padding to each box's own height, capped so tall boxes don't
        # reach across panel borders.
        pad_a = min(int((ay2 - ay1) * gap_ratio), 60)
        pad_b = min(int((by2 - by1) * gap_ratio), 60)
        pad   = (pad_a + pad_b) // 2
        return (
            ax1 - pad < bx2 + pad
            and ax2 + pad > bx1 - pad
            and ay1 - pad < by2 + pad
            and ay2 + pad > by1 - pad
        )

    # ── Union-Find ────────────────────────────────────────────────────────────
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]   # path compression
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    for i in range(n):
        for j in range(i + 1, n):
            if padded_overlap(boxes[i], boxes[j]):
                union(i, j)

    # ── Merge each connected component into one bounding box ──────────────────
    groups: dict = defaultdict(list)
    for i, box in enumerate(boxes):
        groups[find(i)].append(box)

    return [
        (
            min(b[0] for b in g),
            min(b[1] for b in g),
            max(b[2] for b in g),
            max(b[3] for b in g),
        )
        for g in groups.values()
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 – Crop preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_crop(crop_bgr: np.ndarray) -> Image.Image:
    """
    Prepare a detected crop for manga-ocr:
      1. Grayscale conversion
      2. CLAHE — local contrast normalisation (helps thin/faded strokes)
      3. 2× upscale when the crop is small (manga-ocr prefers ≥64 px text)
      4. Light denoising to reduce screen-tone / halftone interference
    """
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    gray = clahe.apply(gray)

    h, w = gray.shape
    if max(h, w) < 600:
        gray = cv2.resize(gray, None, fx=2.0, fy=2.0,
                          interpolation=cv2.INTER_CUBIC)

    gray = cv2.fastNlMeansDenoising(gray, h=5,
                                    templateWindowSize=7,
                                    searchWindowSize=21)

    return Image.fromarray(gray)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3 – OCR output filtering
# ─────────────────────────────────────────────────────────────────────────────

_PUNCT_ONLY = set(
    "。、.,!?！？・…･．ー―-—~～()[]{}<>「」『』【】〔〕"
    " \t\n　／/\\|＿_＊*●◆◇■□▲△▼▽☆★"
)


def is_meaningful(text: str) -> bool:
    """Return True only when the string contains at least one non-punctuation char."""
    s = text.strip()
    return bool(s) and any(ch not in _PUNCT_ONLY for ch in s)


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def save_debug_image(img_bgr: np.ndarray,
                     results: List[Tuple[Box, str]],
                     out_path: str) -> None:
    vis = img_bgr.copy()
    for idx, ((x1, y1, x2, y2), _text) in enumerate(results, start=1):
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 210, 60), 2)
        cv2.putText(vis, str(idx), (x1, max(14, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 210, 60), 2,
                    cv2.LINE_AA)
    cv2.imwrite(out_path, vis)
    print(f"Debug image → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline entry point
# ─────────────────────────────────────────────────────────────────────────────

def run(
    image_path: str,
    debug: bool = False,
    save_crops_dir: str | None = None,
    debug_image: str | None = None,
    no_gpu: bool = False,
) -> None:
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    img_h, img_w = img.shape[:2]

    # ── 1. Detect ─────────────────────────────────────────────────────────────
    print("Detecting text regions (CRAFT)…")
    raw_boxes = detect_text_regions(img, gpu=not no_gpu)

    boxes = filter_boxes(raw_boxes, img_w, img_h)
    boxes = nms(boxes, threshold=0.4)
    boxes = filter_furigana(boxes, img_h)
    boxes = merge_nearby_boxes(boxes, gap_ratio=0.05)
    boxes = sort_manga_order(boxes, img_h)

    if debug:
        print(
            f"[debug] raw={len(raw_boxes)} → "
            f"filter+NMS={len(nms(filter_boxes(raw_boxes, img_w, img_h), 0.4))} → "
            f"drop furigana+merge={len(boxes)}"
        )

    if not boxes:
        print("No text regions detected.")
        return

    # ── 2. Recognise ──────────────────────────────────────────────────────────
    print("Loading manga-ocr…")
    mocr = MangaOcr()

    if save_crops_dir:
        os.makedirs(save_crops_dir, exist_ok=True)

    results: List[Tuple[Box, str]] = []

    for idx, (x1, y1, x2, y2) in enumerate(boxes, start=1):
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        pil_crop = preprocess_crop(crop)

        if save_crops_dir:
            save_path = Path(save_crops_dir) / f"{idx:03d}_{x1}_{y1}.png"
            pil_crop.save(str(save_path))

        text = mocr(pil_crop).strip()

        if debug:
            print(f"[debug] #{idx:02d}  ({x1},{y1})→({x2},{y2})  {text!r}")

        if not is_meaningful(text):
            continue

        results.append(((x1, y1, x2, y2), text))

    # ── 3. Print results ──────────────────────────────────────────────────────
    if not results:
        print("No text extracted.")
        return

    print("\n=== OCR Results ===")
    for i, (box, text) in enumerate(results, start=1):
        x1, y1, x2, y2 = box
        print(f"[{i:02d}]  ({x1:4d},{y1:4d})  {text}")

    if debug_image:
        save_debug_image(img, results, debug_image)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Manga OCR — CRAFT detection + manga-ocr recognition"
    )
    p.add_argument("--image", required=True,
                   help="Path to the manga image or panel")
    p.add_argument("--debug", action="store_true",
                   help="Print per-box detection and recognition details")
    p.add_argument("--save-crops", default=None, metavar="DIR",
                   help="Save preprocessed crops to DIR for manual inspection")
    p.add_argument("--debug-image", default=None, metavar="PATH",
                   help="Write annotated image showing detected boxes to PATH")
    p.add_argument("--no-gpu", action="store_true",
                   help="Force CPU inference (slower but useful without CUDA)")
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    run(
        image_path=args.image,
        debug=args.debug,
        save_crops_dir=args.save_crops,
        debug_image=args.debug_image,
        no_gpu=args.no_gpu,
    )
