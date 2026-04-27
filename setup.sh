#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Motion-Aware Video Segmentation Pipeline — Setup ==="

# ---------- Third-party: SAMURAI only (RAFT is from torchvision) ----------
echo ""
echo "[1/4] Cloning SAMURAI (modified SAM2)..."
mkdir -p third_party

if [ ! -d "third_party/samurai/.git" ]; then
    git clone https://github.com/yangchris11/samurai.git third_party/samurai
else
    echo "  SAMURAI already cloned, skipping."
fi

# ---------- Install SAM2 from SAMURAI ----------
echo ""
echo "[2/4] Installing SAM2 from SAMURAI..."
# SAM2 needs loguru at runtime but doesn't list it in setup.py — install first
pip install -q loguru
if [ -d "third_party/samurai/sam2" ]; then
    pip install -e third_party/samurai/sam2
else
    echo "  WARNING: third_party/samurai/sam2 not found — SAM2 install skipped."
fi

# ---------- Download SAM2 Base+ checkpoint ----------
echo ""
echo "[3/4] Downloading SAM2 Base+ checkpoint..."
mkdir -p checkpoints
SAM2_CKPT="checkpoints/sam2.1_hiera_base_plus.pt"
if [ ! -f "$SAM2_CKPT" ]; then
    wget -q --show-progress -P checkpoints/ \
        https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt
else
    echo "  sam2.1_hiera_base_plus.pt already exists, skipping."
fi

# ---------- Install Python dependencies ----------
echo ""
echo "[4/4] Installing Python dependencies..."
pip install -r requirements.txt

echo ""
echo "=== Setup complete! ==="
echo "Pass 1 uses torchvision's RAFT (weights auto-download on first run)."
echo "Run:  python pipeline.py --config config.yaml --input <video_or_frames>"
