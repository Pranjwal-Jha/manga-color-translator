# Manga Translator & Colorizer

An AI-powered pipeline to automatically detect, erase, translate, and re-render text in manga, with optional AI-assisted coloring.

## Features

- **Text Detection:** Uses EasyOCR (CRAFT) to precisely find speech bubbles and SFX.
- **OCR:** Specialized Japanese recognition via `manga-ocr`.
- **Translation:** Natural translation using local LLMs (Qwen2.5) via `llama.cpp`.
- **Inpainting:** Seamless text removal using `simple-lama-inpainting`.
- **Rendering:** Automatic font fitting (AnimeAce/Roboto) into speech bubbles.
- **Coloring:** AI coloring support (requires external ComfyUI server).

## Examples

### 1. Translation & Typesetting
The pipeline detects Japanese text, removes it using AI inpainting, and renders the English translation in the original style.

| Original | Translated (Final) |
| :--- | :--- |
| ![Original](ui/_work/dc11c794/original.png) | ![Final](ui/_work/dc11c794/final.png) |

### 2. AI Coloring
AI-assisted coloring transforms black-and-white panels into vibrant colored art.

| Original Lineart | AI Colored |
| :--- | :--- |
| ![Lineart](ui/_work/c-1b9fdc/color_original.png) | ![Colored](ui/_work/c-1b9fdc/color_result.png) |

---

## Quick Start (Recommended)

The easiest way to set up the project is using `uv`. We provide scripts for both Linux and Windows.

### Linux / macOS
```bash
chmod +x setup_and_run.sh
./setup_and_run.sh
```

### Windows
```powershell
.\setup_and_run.ps1
```

---

## Manual Setup

If you prefer to set things up manually:

1. **Install uv:** [Installation Guide](https://docs.astral.sh/uv/getting-started/installation/)
2. **Install Dependencies:**
   ```bash
   uv sync
   ```
3. **Run Translation LLM:**
   The server expects a `llama.cpp` server (or compatible OpenAI-like API) on port `8080`.
   ```bash
   ./llama-server -m qwen2.5-7b-instruct.gguf --port 8080
   ```
4. **Start the Web UI:**
   ```bash
   uv run uvicorn ui.server:app --host 0.0.0.0 --port 8000 --reload
   ```
   Open: `http://localhost:8000/static/index.html`

## Configuration

- **LLM_URL:** Set this environment variable to change the translation backend (default: `http://localhost:8080`).
- **GPU Support:** The system automatically uses CUDA if available. On CPU-only systems, it will fall back to standard execution (using `onnxruntime` and `torch` CPU).

## Credits
- Detection: [EasyOCR](https://github.com/JaidedAI/EasyOCR)
- Recognition: [manga-ocr](https://github.com/kha-white/manga-ocr)
- Inpainting: [LaMa](https://github.com/advimman/lama)
- Project Structure: Developed for modern local AI pipelines.
