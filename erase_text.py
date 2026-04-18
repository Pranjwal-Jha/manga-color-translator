#!/usr/bin/env python3
"""
Manga text eraser
=================
Pipeline: CRAFT detection → binary text mask → morphological dilation → LaMa inpainting

Uses simple-lama-inpainting (PyTorch-native) for GPU inference, which works with
any CUDA version that PyTorch supports — no onnxruntime CUDA version mismatch.

Usage
-----
  uv run python erase_text.py --image manga_page.png
  uv run python erase_text.py --image page.jpg --output clean.png --debug
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image
from simple_lama_inpainting import SimpleLama

# Reuse detection pipeline from main.py
from main import (
    Box,
    detect_text_regions,
    filter_boxes,
    nms,
    filter_furigana,
    merge_nearby_boxes,
)




# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — Binary text mask generation
# ─────────────────────────────────────────────────────────────────────────────

def _is_dark_text(crop_gray: np.ndarray) -> bool:
    """
    Determine if text is dark-on-light (common in manga speech bubbles)
    by comparing the mean intensity of a border ring vs the centre.
    """
    h, w = crop_gray.shape
    # Border: outer 20% ring
    bw, bh = max(2, w // 5), max(2, h // 5)
    border_mean = np.mean(
        np.concatenate([
            crop_gray[:bh, :].ravel(),       # top
            crop_gray[-bh:, :].ravel(),       # bottom
            crop_gray[:, :bw].ravel(),        # left
            crop_gray[:, -bw:].ravel(),       # right
        ])
    )
    centre = crop_gray[bh:-bh, bw:-bw] if bh < h // 2 and bw < w // 2 else crop_gray
    centre_mean = np.mean(centre)
    # If the border is brighter than the centre, text is dark
    return border_mean > centre_mean


def generate_text_mask(
    img_bgr: np.ndarray,
    boxes: List[Box],
    dilation_offset: int = 5,
    kernel_size: int = 5,
    debug: bool = False,
) -> np.ndarray:
    """
    Build a binary mask (0/255) covering text pixels for the given boxes.

    For each bounding box:
      1. Extract the crop → grayscale
      2. Adaptive threshold to segment text from background
      3. Determine text polarity (dark-on-light vs light-on-dark)
      4. Fill the crop mask into the full-image mask
      5. Morphological dilation to cover stroke edges + margin

    Returns a single-channel uint8 mask (same size as input image).
    """
    img_h, img_w = img_bgr.shape[:2]
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    gray_full = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    for x1, y1, x2, y2 in boxes:
        crop_gray = gray_full[y1:y2, x1:x2]
        if crop_gray.size == 0:
            continue

        bw, bh = x2 - x1, y2 - y1

        # Adaptive Gaussian threshold — works well on varying background
        # Block size must be odd and > 1; scale to crop size
        block_size = max(3, (min(bw, bh) // 4) | 1)  # ensure odd
        thresh = cv2.adaptiveThreshold(
            crop_gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=block_size,
            C=8,
        )

        # If text is light-on-dark, invert the threshold
        if not _is_dark_text(crop_gray):
            thresh = cv2.bitwise_not(thresh)

        # Remove tiny noise blobs via morphological opening
        noise_kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, noise_kern)

        # Paste into full mask
        mask[y1:y2, x1:x2] = cv2.bitwise_or(mask[y1:y2, x1:x2], thresh)

    # ── Global dilation ──────────────────────────────────────────────────────
    # Scale dilation kernel to approximate manga-image-translator's formula:
    #   dilate_size = max((int((text_size + offset) * 0.3) // 2) * 2 + 1, 3)
    # We use a simpler global kernel since we already have per-box masks.
    dilate_size = max(kernel_size, 3)
    if dilate_size % 2 == 0:
        dilate_size += 1
    kern = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (dilate_size, dilate_size)
    )
    mask = cv2.dilate(mask, kern, iterations=2)

    # Additional dilation pass with the offset-based kernel for better coverage
    if dilation_offset > 0:
        offset_size = max((dilation_offset // 2) * 2 + 1, 3)
        offset_kern = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (offset_size, offset_size)
        )
        mask = cv2.dilate(mask, offset_kern, iterations=1)

    return mask


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3 — LaMa inpainting (PyTorch-native, CUDA 13 compatible)
# ─────────────────────────────────────────────────────────────────────────────

def lama_inpaint(
    img_rgb: np.ndarray,
    mask: np.ndarray,
    boxes: List[Box],
    model: SimpleLama,
) -> np.ndarray:
    """
    Run per-region LaMa inpainting for higher quality.

    Instead of shrinking the entire image to 512×512 (which loses detail),
    we inpaint each detected text region individually with generous context
    padding so LaMa sees enough surrounding information to fill naturally.

    Args:
        img_rgb:  H×W×3  uint8  RGB image
        mask:     H×W    uint8  binary mask (255 = inpaint region)
        boxes:    list of (x1, y1, x2, y2) bounding boxes
        model:    SimpleLama instance (runs on GPU if available)

    Returns:
        H×W×3  uint8  RGB inpainted image
    """
    output = img_rgb.copy()
    img_h, img_w = img_rgb.shape[:2]

    for x1, y1, x2, y2 in boxes:
        bw, bh = x2 - x1, y2 - y1

        # Skip if there are no mask pixels in this box
        if mask[y1:y2, x1:x2].max() == 0:
            continue

        # Generous padding: 50% of box size on each side for context
        pad = max(max(bw, bh) // 2, 50)
        px1 = max(0, x1 - pad)
        py1 = max(0, y1 - pad)
        px2 = min(img_w, x2 + pad)
        py2 = min(img_h, y2 + pad)

        tile_img  = output[py1:py2, px1:px2].copy()
        tile_mask = mask[py1:py2, px1:px2].copy()

        # simple-lama-inpainting expects PIL Images
        pil_img  = Image.fromarray(tile_img)
        pil_mask = Image.fromarray(tile_mask)

        inpainted_pil  = model(pil_img, pil_mask)
        inpainted_tile = np.array(inpainted_pil)

        # SimpleLama may return a slightly different size due to internal
        # 512×512 resize + back-resize rounding — snap it to tile dims.
        th, tw = tile_img.shape[:2]
        if inpainted_tile.shape[:2] != (th, tw):
            inpainted_tile = cv2.resize(inpainted_tile, (tw, th),
                                        interpolation=cv2.INTER_LINEAR)

        # Composite: only paste into masked pixels
        tile_mask_bool = (tile_mask > 127)[:, :, np.newaxis]
        output[py1:py2, px1:px2] = np.where(
            tile_mask_bool, inpainted_tile, output[py1:py2, px1:px2]
        )

    return output


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run(
    image_path: str,
    output_path: str = "erased.png",
    debug: bool = False,
    save_mask: str | None = None,
    no_gpu: bool = False,
    dilation_offset: int = 5,
    kernel_size: int = 5,
) -> None:
    img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    img_h, img_w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # ── 1. Detect text regions (CRAFT) ───────────────────────────────────────
    print("Stage 1: Detecting text regions (CRAFT)…")
    raw_boxes = detect_text_regions(img_bgr, gpu=not no_gpu)
    boxes = filter_boxes(raw_boxes, img_w, img_h)
    boxes = nms(boxes, threshold=0.4)
    boxes = filter_furigana(boxes, img_h)
    boxes = merge_nearby_boxes(boxes, gap_ratio=0.15)

    if not boxes:
        print("No text regions detected — nothing to erase.")
        cv2.imwrite(output_path, img_bgr)
        return

    print(f"  Found {len(boxes)} text region(s).")

    # ── 2. Generate binary mask ──────────────────────────────────────────────
    print("Stage 2: Generating text mask…")
    mask = generate_text_mask(
        img_bgr, boxes,
        dilation_offset=dilation_offset,
        kernel_size=kernel_size,
        debug=debug,
    )

    if save_mask:
        cv2.imwrite(save_mask, mask)
        print(f"  Mask saved → {save_mask}")

    if debug:
        # Save a debug overlay showing mask on top of image
        overlay = img_bgr.copy()
        overlay[mask > 127] = (0, 0, 255)  # red tint on masked areas
        blended = cv2.addWeighted(img_bgr, 0.5, overlay, 0.5, 0)
        debug_path = str(Path(output_path).with_suffix("")) + "_mask_overlay.png"
        cv2.imwrite(debug_path, blended)
        print(f"  Mask overlay → {debug_path}")

    # ── 3. LaMa inpainting ───────────────────────────────────────────────────
    print("Stage 3: LaMa inpainting (tiled)…")
    model = SimpleLama()  # downloads model on first run, auto-selects CUDA if available
    result_rgb = lama_inpaint(img_rgb, mask, boxes, model)

    # ── Save result ──────────────────────────────────────────────────────────
    result_bgr = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, result_bgr)
    print(f"Done! Text-erased image → {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Manga text eraser — CRAFT + binary mask + dilation + LaMa inpainting"
    )
    p.add_argument("--image", required=True,
                   help="Path to the manga image")
    p.add_argument("--output", default="erased.png",
                   help="Output path for the text-erased image (default: erased.png)")
    p.add_argument("--debug", action="store_true",
                   help="Save debug visualisations (mask overlay, etc.)")
    p.add_argument("--save-mask", default=None, metavar="PATH",
                   help="Save the dilated binary mask to PATH")
    p.add_argument("--no-gpu", action="store_true",
                   help="Force CPU inference")
    p.add_argument("--dilation-offset", type=int, default=14,
                   help="Extra dilation applied to the mask (default: 14)")
    p.add_argument("--kernel-size", type=int, default=9,
                   help="Base dilation kernel size, must be odd (default: 9)")
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    run(
        image_path=args.image,
        output_path=args.output,
        debug=args.debug,
        save_mask=args.save_mask,
        no_gpu=args.no_gpu,
        dilation_offset=args.dilation_offset,
        kernel_size=args.kernel_size,
    )
