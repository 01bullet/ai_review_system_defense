#!/usr/bin/env bash
set -e

echo "============================================================"
echo "  AI Review System — 论文审稿与攻击防御"
echo "============================================================"
echo ""

# Check Python
if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
    echo "[ERROR] Python not found. Please install Python 3.10+ first."
    exit 1
fi

PYTHON=$(command -v python3 || command -v python)

# Install requirements (if needed)
if [ ! -f ".requirements_installed" ]; then
    echo "[1/3] Installing Python dependencies..."
    $PYTHON -m pip install -r requirements.txt
    touch ".requirements_installed"
    echo "[OK] Dependencies installed."
else
    echo "[1/3] Dependencies already installed."
fi

# Ensure base model downloaded
echo ""
echo "[2/3] Checking base model (Qwen2.5-7B ~15GB, downloaded on first run)..."
$PYTHON ensure_model.py

# Start server
echo ""
echo "[3/3] Starting AI Review System..."
echo ""
echo "  Open http://localhost:8000 in your browser"
echo "  Press Ctrl+C to stop"
echo ""
$PYTHON review_app.py
