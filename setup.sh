#!/usr/bin/env bash
# ============================================================
# Real Estate Room Classifier — Environment Setup
# ============================================================
# Usage:
#   1. Upload project_data.zip to Google Drive
#   2. Get the file ID from the share link:
#      https://drive.google.com/file/d/<FILE_ID>/view
#   3. Run:
#      bash setup.sh <GOOGLE_DRIVE_FILE_ID>
# ============================================================

set -euo pipefail

FILE_ID="${1:-}"
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"
ZIP_NAME="project_data.zip"

if [ -z "$FILE_ID" ]; then
    echo "Usage: bash setup.sh <GOOGLE_DRIVE_FILE_ID>"
    echo ""
    echo "To get the file ID:"
    echo "  1. Upload project_data.zip to Google Drive"
    echo "  2. Right-click → Share → Copy link"
    echo "  3. Extract ID from: https://drive.google.com/file/d/<FILE_ID>/view"
    exit 1
fi

echo "============================================"
echo " Real Estate Room Classifier — Setup"
echo "============================================"

# ── 1. Download from Google Drive ────────────────────────────────
echo ""
echo "[1/5] Downloading from Google Drive..."
if command -v gdown &> /dev/null; then
    gdown "https://drive.google.com/uc?id=$FILE_ID" -O "$PROJECT_DIR/$ZIP_NAME"
else
    echo "Installing gdown..."
    pip install gdown -q
    gdown "https://drive.google.com/uc?id=$FILE_ID" -O "$PROJECT_DIR/$ZIP_NAME"
fi

# ── 2. Extract ZIP ──────────────────────────────────────────────
echo ""
echo "[2/5] Extracting data..."
unzip -q -o "$PROJECT_DIR/$ZIP_NAME" -d "$PROJECT_DIR"
rm -f "$PROJECT_DIR/$ZIP_NAME"
echo "  ✓ Extracted to $PROJECT_DIR"

# ── 3. Create virtual environment ───────────────────────────────
echo ""
echo "[3/5] Setting up Python virtual environment..."
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
    echo "  ✓ Created venv at $VENV_DIR"
else
    echo "  ✓ venv already exists"
fi

source "$VENV_DIR/bin/activate"

# ── 4. Install dependencies ─────────────────────────────────────
echo ""
echo "[4/5] Installing dependencies..."
pip install --upgrade pip -q
pip install -r "$PROJECT_DIR/requirements.txt" -q
echo "  ✓ Dependencies installed"

# ── 5. Verify setup ─────────────────────────────────────────────
echo ""
echo "[5/5] Verifying setup..."
python -c "
import torch
import clip
import xgboost
import streamlit
from src.labels import ROOM_LABELS

print(f'  ✓ PyTorch {torch.__version__} (CUDA: {torch.cuda.is_available()})')
print(f'  ✓ CLIP loaded')
print(f'  ✓ XGBoost {xgboost.__version__}')
print(f'  ✓ Streamlit {streamlit.__version__}')
print(f'  ✓ {len(ROOM_LABELS)} room labels loaded')

import joblib
from pathlib import Path
models_dir = Path('$PROJECT_DIR/outputs/comparison/models')
for model_name in ['clip', 'dinov2', 'place365']:
    model_path = models_dir / model_name / 'best_classifier.joblib'
    if model_path.exists():
        clf = joblib.load(model_path)
        print(f'  ✓ {model_name.upper()} classifier loaded ({type(clf).__name__})')
    else:
        print(f'  ✗ {model_name.upper()} classifier NOT FOUND at {model_path}')
"

echo ""
echo "============================================"
echo " Setup complete!"
echo "============================================"
echo ""
echo "To run the Streamlit app:"
echo "  source .venv/bin/activate"
echo "  streamlit run src/web/app.py"
echo ""
echo "To train models:"
echo "  source .venv/bin/activate"
echo "  python -m src.ml.compare_models"
echo ""
