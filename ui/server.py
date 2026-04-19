#!/usr/bin/env python3
"""
Manga Translation Web API
=========================
FastAPI server that orchestrates the full pipeline:
  1. CRAFT text detection
  2. manga-ocr recognition
  3. Local LLM translation (Qwen via llama.cpp on :8080)
  4. LaMa text erasure (inpainting)
  5. Translated text rendering

Run:
  cd /path/to/cv_proj
  uv run uvicorn ui.server:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

# Use cached HuggingFace models — avoids SSL errors on systems with
# broken OpenSSL and speeds up startup since no network check is needed.
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import asyncio
import json
import os
import sys
import time
import traceback
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import requests

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageDraw, ImageFont

# ── Ensure project root is importable ────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from main import (
    Box,
    detect_text_regions,
    filter_boxes,
    filter_furigana,
    merge_nearby_boxes,
    nms,
    preprocess_crop,
    sort_manga_order,
    is_meaningful,
)
from erase_text import generate_text_mask, lama_inpaint
from render_text import (
    get_bubble_bounds,
    get_font_path,
    fit_text,
)

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(title="Manga Translator")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static frontend
UI_DIR = Path(__file__).resolve().parent
app.mount("/static", StaticFiles(directory=str(UI_DIR / "static")), name="static")

# Directory for temporary processing outputs
WORK_DIR = PROJECT_ROOT / "ui" / "_work"
WORK_DIR.mkdir(parents=True, exist_ok=True)

# ── In-memory job store ───────────────────────────────────────────────────────
jobs: Dict[str, dict] = {}


# ── Lazy-loaded heavy models (singleton) ─────────────────────────────────────
_models: dict = {}


def _get_mocr():
    if "mocr" not in _models:
        from manga_ocr import MangaOcr
        _models["mocr"] = MangaOcr()
    return _models["mocr"]


def _get_lama():
    if "lama" not in _models:
        from simple_lama_inpainting import SimpleLama
        _models["lama"] = SimpleLama()
    return _models["lama"]


# ── Local LLM translation via llama.cpp ──────────────────────────────────────
LLM_URL = os.environ.get("LLM_URL", "http://localhost:8080")

_TRANSLATE_SYSTEM_PROMPT = """You are a professional Japanese-to-English manga translator.
You translate speech bubbles, sound effects, and narration from manga panels.

Rules:
- Translate naturally — use casual, punchy English that fits manga tone.
- Preserve onomatopoeia/SFX as expressive English equivalents (e.g. ドドド → *RUMBLE*, バーン → BANG!!).
- Keep translations SHORT — these must fit inside small speech bubbles.
- For very short exclamations (えっ, あっ, おい), use natural English equivalents (Huh?!, Ah!, Hey!).
- Do NOT add explanations, notes, or anything besides the translation.
- Return ONLY a JSON array of translated strings, one per input line, in the same order."""


def _translate_batch(texts: List[str]) -> List[str]:
    """
    Translate a batch of Japanese texts using the local llama.cpp LLM.
    Sends all texts in one prompt for efficiency and context awareness.
    """
    numbered = "\n".join(f"{i+1}. {t}" for i, t in enumerate(texts))
    user_msg = (
        f"Translate these {len(texts)} Japanese manga speech bubbles to English.\n"
        f"Return a JSON array with exactly {len(texts)} translated strings.\n\n"
        f"{numbered}"
    )

    try:
        resp = requests.post(
            f"{LLM_URL}/v1/chat/completions",
            json={
                "messages": [
                    {"role": "system", "content": _TRANSLATE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                "temperature": 0.3,
                "max_tokens": 2048,
            },
            timeout=120,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"].strip()

        # Parse the JSON array from the response
        # Handle cases where LLM wraps in ```json ... ```
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()

        translations = json.loads(content)

        if isinstance(translations, list) and len(translations) == len(texts):
            return [str(t) for t in translations]

        # Fallback: if wrong count, try line-by-line
        print(f"[warn] LLM returned {len(translations)} items for {len(texts)} inputs")
        # Pad or truncate
        result = [str(t) for t in translations[:len(texts)]]
        while len(result) < len(texts):
            result.append(texts[len(result)])  # keep original as fallback
        return result

    except (requests.RequestException, json.JSONDecodeError, KeyError, IndexError) as e:
        print(f"[error] LLM translation failed: {e}")
        # Fallback: translate one-by-one
        return _translate_individual(texts)


def _translate_individual(texts: List[str]) -> List[str]:
    """Fallback: translate texts one at a time."""
    results = []
    for text in texts:
        try:
            resp = requests.post(
                f"{LLM_URL}/v1/chat/completions",
                json={
                    "messages": [
                        {"role": "system", "content": _TRANSLATE_SYSTEM_PROMPT},
                        {"role": "user", "content": f"Translate this Japanese manga text to English (reply with ONLY the translation, nothing else):\n{text}"},
                    ],
                    "temperature": 0.3,
                    "max_tokens": 256,
                },
                timeout=30,
            )
            resp.raise_for_status()
            en = resp.json()["choices"][0]["message"]["content"].strip()
            # Strip quotes if LLM wrapped it
            if en.startswith('"') and en.endswith('"'):
                en = en[1:-1]
            results.append(en if en else text)
        except Exception:
            results.append(text)
    return results


# ── Helpers ──────────────────────────────────────────────────────────────────

def _save_image(img_rgb: np.ndarray, path: Path) -> None:
    """Save an RGB numpy array as PNG."""
    Image.fromarray(img_rgb).save(str(path))


def _draw_boxes_debug(img_rgb: np.ndarray, boxes: List[Box]) -> np.ndarray:
    """Draw green numbered bounding boxes on image for debug view."""
    vis = img_rgb.copy()
    for idx, (x1, y1, x2, y2) in enumerate(boxes, start=1):
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 210, 60), 2)
        cv2.putText(
            vis, str(idx), (x1, max(14, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 210, 60), 2, cv2.LINE_AA,
        )
    return vis


def _draw_mask_overlay(img_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Red semi-transparent overlay on masked regions."""
    overlay = img_rgb.copy()
    overlay[mask > 127] = [255, 60, 60]
    return cv2.addWeighted(img_rgb, 0.55, overlay, 0.45, 0)


def _render_translated_text(
    erased_rgb: np.ndarray,
    img_bgr_original: np.ndarray,
    boxes: List[Box],
    translations: List[str],
) -> np.ndarray:
    """Render translated English text onto the erased image."""
    font_path = get_font_path()
    target = Image.fromarray(erased_rgb)
    draw = ImageDraw.Draw(target)

    for box, text in zip(boxes, translations):
        ox1, oy1, ox2, oy2 = box

        # Bubble extraction
        bx1, by1, bx2, by2 = get_bubble_bounds(img_bgr_original, ox1, oy1, ox2, oy2)
        bw = bx2 - bx1
        bh = by2 - by1

        # Fallback
        obw = ox2 - ox1
        obh = oy2 - oy1
        if bw < obw or bh < obh:
            best_size = max(obw, obh)
            cx, cy = (ox1 + ox2) // 2, (oy1 + oy2) // 2
            bw, bh = best_size, best_size
            bx1, by1 = cx - bw // 2, cy - bh // 2
            bx2, by2 = bx1 + bw, by1 + bh

        font, lines, font_size = fit_text(text, bw, bh, font_path)
        ascent, descent = font.getmetrics()
        line_height = ascent + descent
        spacing = font_size * 0.15
        total_h = len(lines) * line_height + max(0, len(lines) - 1) * spacing
        current_y = by1 + (bh - total_h) / 2
        stroke_width = max(1, int(font_size * 0.05))

        for line in lines:
            line_w = font.getlength(line)
            current_x = bx1 + (bw - line_w) / 2
            draw.text(
                (current_x, current_y), line,
                fill="black", font=font,
                stroke_width=stroke_width, stroke_fill="white",
            )
            current_y += line_height + spacing

    return np.array(target)


# ── Pipeline (runs in thread) ────────────────────────────────────────────────

def _run_pipeline(job_id: str, img_bytes: bytes) -> None:
    """Full end-to-end pipeline. Updates jobs[job_id] with progress."""
    job = jobs[job_id]
    job_dir = WORK_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    try:
        # ── Decode image ─────────────────────────────────────────────────
        arr = np.frombuffer(img_bytes, np.uint8)
        img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError("Failed to decode uploaded image")
        img_h, img_w = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Save original
        _save_image(img_rgb, job_dir / "original.png")
        job["status"] = "detecting"
        job["progress"] = 10

        # ── 1. Detection ──────────────────────────────────────────────────
        import torch
        raw_boxes = detect_text_regions(img_bgr, gpu=torch.cuda.is_available())
        boxes = filter_boxes(raw_boxes, img_w, img_h)
        boxes = nms(boxes, threshold=0.4)
        boxes = filter_furigana(boxes, img_h)
        boxes = merge_nearby_boxes(boxes, gap_ratio=0.05)
        boxes = sort_manga_order(boxes, img_h)

        if not boxes:
            job["status"] = "done"
            job["progress"] = 100
            job["result"] = "No text detected in this image."
            _save_image(img_rgb, job_dir / "final.png")
            return

        # Save detection debug
        det_vis = _draw_boxes_debug(img_rgb, boxes)
        _save_image(det_vis, job_dir / "detection.png")
        job["status"] = "recognizing"
        job["progress"] = 25
        job["num_regions"] = len(boxes)

        # ── 2. OCR ────────────────────────────────────────────────────────
        mocr = _get_mocr()
        ocr_results: List[Tuple[Box, str]] = []

        for idx, (x1, y1, x2, y2) in enumerate(boxes):
            crop = img_bgr[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            pil_crop = preprocess_crop(crop)
            text = mocr(pil_crop).strip()
            if is_meaningful(text):
                ocr_results.append(((x1, y1, x2, y2), text))

        if not ocr_results:
            job["status"] = "done"
            job["progress"] = 100
            job["result"] = "Text detected but OCR returned no meaningful results."
            _save_image(img_rgb, job_dir / "final.png")
            return

        job["ocr_texts"] = [t for _, t in ocr_results]
        job["boxes"] = [list(b) for b, _ in ocr_results]
        job["status"] = "translating"
        job["progress"] = 40

        # ── 3. Translation (local LLM via llama.cpp) ────────────────────
        jp_texts = [t for _, t in ocr_results]
        translations = _translate_batch(jp_texts)

        job["translations"] = translations
        job["status"] = "erasing"
        job["progress"] = 55

        # ── 4. Text erasure (mask + LaMa) ─────────────────────────────────
        final_boxes = [b for b, _ in ocr_results]

        mask = generate_text_mask(
            img_bgr, final_boxes,
            dilation_offset=14, kernel_size=9,
        )

        # Save mask debug
        mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        _save_image(mask_rgb, job_dir / "mask.png")

        # Save mask overlay
        overlay = _draw_mask_overlay(img_rgb, mask)
        _save_image(overlay, job_dir / "mask_overlay.png")

        job["progress"] = 65

        lama = _get_lama()
        erased_rgb = lama_inpaint(img_rgb, mask, final_boxes, lama)
        _save_image(erased_rgb, job_dir / "erased.png")

        job["status"] = "rendering"
        job["progress"] = 80

        # ── 5. Text rendering ─────────────────────────────────────────────
        final_rgb = _render_translated_text(
            erased_rgb, img_bgr, final_boxes, translations,
        )
        _save_image(final_rgb, job_dir / "final.png")

        job["status"] = "done"
        job["progress"] = 100

    except Exception as e:
        job["status"] = "error"
        job["error"] = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
async def index():
    return FileResponse(str(UI_DIR / "static" / "index.html"))


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    """Upload a manga panel and start the translation pipeline."""
    job_id = str(uuid.uuid4())[:8]
    img_bytes = await file.read()

    jobs[job_id] = {
        "status": "queued",
        "progress": 0,
        "created": time.time(),
    }

    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, _run_pipeline, job_id, img_bytes)

    return {"job_id": job_id}


@app.get("/api/status/{job_id}")
async def status(job_id: str):
    """Poll the status of a translation job."""
    if job_id not in jobs:
        return JSONResponse(status_code=404, content={"error": "Job not found"})
    return jobs[job_id]


@app.get("/api/image/{job_id}/{stage}")
async def get_image(job_id: str, stage: str):
    """
    Serve an intermediate or final image.
    Stages: original, detection, mask, mask_overlay, erased, final
    """
    allowed = {"original", "detection", "mask", "mask_overlay", "erased", "final"}
    if stage not in allowed:
        return JSONResponse(status_code=400, content={"error": f"Invalid stage: {stage}"})
    path = WORK_DIR / job_id / f"{stage}.png"
    if not path.exists():
        return JSONResponse(status_code=404, content={"error": "Image not ready yet"})
    return FileResponse(str(path), media_type="image/png")


# ── Model VRAM management ────────────────────────────────────────────────────

@app.post("/api/unload_models")
async def unload_models():
    """Free GPU VRAM by unloading cached models (manga-ocr, LaMa, EasyOCR) and stopping llama server."""
    import gc
    import subprocess

    unloaded = []
    for name in list(_models.keys()):
        del _models[name]
        unloaded.append(name)

    # Try to stop the llama.cpp docker container
    try:
        result = subprocess.run(
            ["docker", "ps", "-q", "--filter", "ancestor=local/llama.cpp:server-cuda"],
            capture_output=True, text=True, check=True
        )
        for cid in result.stdout.strip().split("\n"):
            if cid:
                subprocess.run(["docker", "stop", "-t", "2", cid], check=False)
                unloaded.append("llama-server")
    except Exception as e:
        print(f"Could not stop llama container: {e}")

    gc.collect()

    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            free, total = torch.cuda.mem_get_info()
            vram_free_mb = free / 1024 / 1024
            vram_total_mb = total / 1024 / 1024
            return {
                "unloaded": unloaded,
                "vram_free_mb": round(vram_free_mb),
                "vram_total_mb": round(vram_total_mb),
            }
    except ImportError:
        pass

    return {"unloaded": unloaded, "vram_free_mb": None}


# ── Colorization (ComfyUI proxy) ─────────────────────────────────────────────

COMFYUI_URL = os.environ.get("COMFYUI_URL", "http://localhost:8188")
WORKFLOW_PATH = UI_DIR / "manga_color_api_cv_lab.json"

color_jobs: Dict[str, dict] = {}


def _upload_to_comfyui(img_bytes: bytes, filename: str) -> str:
    """Upload an image to ComfyUI's input folder and return the server filename."""
    resp = requests.post(
        f"{COMFYUI_URL}/upload/image",
        files={"image": (filename, img_bytes, "image/png")},
        data={"overwrite": "true"},
    )
    resp.raise_for_status()
    return resp.json()["name"]


def _run_colorize(job_id: str, panel_bytes: bytes,
                  ref1_bytes: bytes, ref2_bytes: Optional[bytes]) -> None:
    """Submit the colorization workflow to ComfyUI and poll for result."""
    job = color_jobs[job_id]
    job_dir = WORK_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    try:
        job["status"] = "uploading"
        job["progress"] = 10

        # Save original locally
        arr = np.frombuffer(panel_bytes, np.uint8)
        img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img_bgr is not None:
            _save_image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB),
                        job_dir / "color_original.png")

        # Upload images to ComfyUI
        panel_name = _upload_to_comfyui(panel_bytes, f"{job_id}_panel.png")
        ref1_name = _upload_to_comfyui(ref1_bytes, f"{job_id}_ref1.png")
        ref2_name = ref1_name
        if ref2_bytes:
            ref2_name = _upload_to_comfyui(ref2_bytes, f"{job_id}_ref2.png")

        job["status"] = "queued"
        job["progress"] = 25

        # Build workflow from template
        import copy
        workflow = json.loads(WORKFLOW_PATH.read_text())

        # Node 76 = manga panel, Node 81 = ref1, Node 194 = ref2
        workflow["76"]["inputs"]["image"] = panel_name
        workflow["81"]["inputs"]["image"] = ref1_name
        workflow["194"]["inputs"]["image"] = ref2_name

        # Submit to ComfyUI
        prompt_resp = requests.post(
            f"{COMFYUI_URL}/prompt",
            json={"prompt": workflow},
            timeout=30,
        )
        prompt_resp.raise_for_status()
        prompt_id = prompt_resp.json()["prompt_id"]

        job["status"] = "generating"
        job["progress"] = 40
        job["prompt_id"] = prompt_id

        # Poll ComfyUI history for completion
        for _ in range(600):  # up to ~10 min
            time.sleep(1)
            try:
                hist_resp = requests.get(
                    f"{COMFYUI_URL}/history/{prompt_id}", timeout=10,
                )
                hist = hist_resp.json()
                if prompt_id in hist:
                    outputs = hist[prompt_id].get("outputs", {})
                    # Node 110 is the SaveImage node
                    if "110" in outputs and outputs["110"].get("images"):
                        img_info = outputs["110"]["images"][0]
                        img_url = (
                            f"{COMFYUI_URL}/view?"
                            f"filename={img_info['filename']}"
                            f"&subfolder={img_info.get('subfolder', '')}"
                            f"&type={img_info.get('type', 'output')}"
                        )
                        img_resp = requests.get(img_url, timeout=30)
                        img_resp.raise_for_status()

                        # Save result
                        result_path = job_dir / "color_result.png"
                        result_path.write_bytes(img_resp.content)

                        job["status"] = "done"
                        job["progress"] = 100
                        return

                    # Check for errors
                    status_data = hist[prompt_id].get("status", {})
                    if status_data.get("status_str") == "error":
                        job["status"] = "error"
                        job["error"] = "ComfyUI workflow failed"
                        return
            except requests.RequestException:
                pass  # ComfyUI might be busy, keep polling

            # Update progress linearly while generating
            job["progress"] = min(90, 40 + _ // 6)

        job["status"] = "error"
        job["error"] = "Timeout waiting for ComfyUI to finish"

    except Exception as e:
        job["status"] = "error"
        job["error"] = f"{type(e).__name__}: {e}"


@app.post("/api/colorize")
async def colorize(
    panel: UploadFile = File(...),
    ref1: UploadFile = File(...),
    ref2: Optional[UploadFile] = File(None),
):
    """Upload a B&W manga panel + 1-2 color reference images for colorization."""
    job_id = "c-" + str(uuid.uuid4())[:6]
    panel_bytes = await panel.read()
    ref1_bytes = await ref1.read()
    ref2_bytes = await ref2.read() if ref2 else None

    color_jobs[job_id] = {
        "status": "queued",
        "progress": 0,
        "created": time.time(),
    }

    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, _run_colorize, job_id, panel_bytes,
                         ref1_bytes, ref2_bytes)

    return {"job_id": job_id}


@app.get("/api/colorize/status/{job_id}")
async def colorize_status(job_id: str):
    if job_id not in color_jobs:
        return JSONResponse(status_code=404, content={"error": "Job not found"})
    return color_jobs[job_id]


@app.get("/api/colorize/image/{job_id}/{stage}")
async def colorize_image(job_id: str, stage: str):
    allowed = {"color_original", "color_result"}
    if stage not in allowed:
        return JSONResponse(status_code=400, content={"error": f"Invalid stage"})
    path = WORK_DIR / job_id / f"{stage}.png"
    if not path.exists():
        return JSONResponse(status_code=404, content={"error": "Not ready"})
    return FileResponse(str(path), media_type="image/png")
