#!/bin/bash
set -e

# --- Configuration ---
PORT=8000
LLM_PORT=8080
PROJECT_DIR=$(pwd)

echo "🚀 Starting Manga Translator setup..."

# 1. Check for 'uv'
if ! command -v uv &> /dev/null; then
    echo "❌ 'uv' is not installed. Please install it first: https://docs.astral.sh/uv/getting-started/installation/"
    exit 1
fi

# 2. Sync dependencies and create .venv
echo "📦 Syncing Python environment (using project pyproject.toml)..."
uv sync

# 3. Pre-download core models (OCR, Detection, Inpainting)
# This prevents the UI from hanging on the first request.
echo "📥 Pre-downloading AI models (this may take a few minutes)..."
uv run python3 <<EOF
import os
import torch
from manga_ocr import MangaOcr
from easyocr import Reader
from simple_lama_inpainting import SimpleLama

print("--- Loading MangaOCR (Recognition) ---")
MangaOcr()

print("--- Loading EasyOCR (CRAFT Detection) ---")
Reader(['ja'], gpu=torch.cuda.is_available())

print("--- Loading Simple-LaMa (Inpainting) ---")
SimpleLama()

print("✅ All core models are ready.")
EOF

# 4. Final Instructions
echo ""
echo "===================================================================="
echo "✅ Setup Complete!"
echo ""
echo "⚠️  IMPORTANT: TRANSLATION SERVICE"
echo "This server expects a translation LLM (Qwen) running on port $LLM_PORT."
echo "If you haven't yet, run your translation engine (e.g., llama.cpp):"
echo ""
echo "  # Example for llama.cpp (server mode):"
echo "  ./llama-server -m models/qwen2.5-7b-instruct.gguf --port $LLM_PORT"
echo ""
echo "===================================================================="
echo "▶️  To start the web interface, run:"
echo ""
echo "  uv run uvicorn ui.server:app --host 0.0.0.1 --port $PORT --reload"
echo ""
echo "Then open: http://localhost:$PORT/static/index.html"
echo "===================================================================="
echo ""
