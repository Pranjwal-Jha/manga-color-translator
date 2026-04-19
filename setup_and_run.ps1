$ErrorActionPreference = "Stop"

Write-Host "🚀 Starting Manga Translator Windows setup..." -ForegroundColor Cyan

# 1. Check for 'uv'
if (!(Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "❌ 'uv' is not installed. Please install it first: https://docs.astral.sh/uv/getting-started/installation/" -ForegroundColor Red
    exit 1
}

# 2. Sync dependencies
Write-Host "📦 Syncing Python environment (using uv sync)..." -ForegroundColor Yellow
uv sync

# 3. Pre-download models
Write-Host "📥 Pre-downloading AI models (this may take a few minutes)..." -ForegroundColor Yellow
uv run python -c @"
import torch
from manga_ocr import MangaOcr
from easyocr import Reader
from simple_lama_inpainting import SimpleLama

print('--- Loading MangaOCR ---')
MangaOcr()

print('--- Loading EasyOCR (CRAFT) ---')
Reader(['ja'], gpu=torch.cuda.is_available())

print('--- Loading Simple-LaMa ---')
SimpleLama()

print('✅ All core models are ready.')
"@

# 4. Final Instructions
Write-Host "`n====================================================================" -ForegroundColor Green
Write-Host "✅ Setup Complete!" -ForegroundColor Green
Write-Host ""
Write-Host "⚠️  IMPORTANT: TRANSLATION SERVICE"
Write-Host "This server expects a translation LLM (Qwen 3.5) running on port 8080."
Write-Host "1. Download 'llama-server.exe' (binaries) from llama.cpp GitHub releases."
Write-Host "2. Download the model:"
Write-Host "   curl.exe -L 'https://huggingface.co/unsloth/Qwen3.5-2B-GGUF/resolve/main/Qwen3.5-2B-Q4_K_M.gguf' -o qwen3.5.gguf"
Write-Host "3. Start the server:"
Write-Host "   .\llama-server.exe -m qwen3.5.gguf --port 8080"
Write-Host "===================================================================="
Write-Host "▶️  To start the web interface, run:"
Write-Host ""
Write-Host "  uv run uvicorn ui.server:app --host 0.0.0.0 --port 8000 --reload"
Write-Host ""
Write-Host "Then open: http://localhost:8000/static/index.html"
Write-Host "===================================================================="
