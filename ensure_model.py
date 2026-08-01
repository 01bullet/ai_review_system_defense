"""
Auto-download Qwen2.5-7B base model on first run.

Checks if the base model exists locally; if not, downloads from
HuggingFace mirror (China-friendly) or official HuggingFace.

Usage:
    python ensure_model.py                          # download to default path
    python ensure_model.py --output models/qwen2.5-7b
    python ensure_model.py --source modelscope       # use ModelScope instead

As a module:
    from ensure_model import ensure_base_model
    path = ensure_base_model()  # returns path, downloads if needed
"""

import os
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = PROJECT / "models" / "qwen2.5-7b"

# Mirror endpoints (tried in order)
HF_MIRRORS = [
    ("https://hf-mirror.com", "HF-Mirror (China)"),
    ("https://hf.xeduapi.com", "XeduAPI Mirror"),
    ("https://huggingface.co", "HuggingFace Official"),
]


def _test_endpoint(endpoint: str, timeout: float = 5.0) -> bool:
    """Quick connectivity test."""
    import urllib.request
    try:
        req = urllib.request.Request(
            f"{endpoint}/api/models/Qwen/Qwen2.5-7B",
            headers={"User-Agent": "ensure_model/1.0"},
        )
        urllib.request.urlopen(req, timeout=timeout)
        return True
    except Exception:
        return False


def _download_from_hf_hub(
    model_id: str,
    output_dir: Path,
    endpoint: str,
    resume: bool = True,
) -> bool:
    """Download using huggingface_hub. Returns True on success."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("  [ERROR] huggingface_hub not installed. Run: pip install huggingface_hub")
        return False

    os.environ["HF_ENDPOINT"] = endpoint
    print(f"  Downloading {model_id} ...")
    print(f"  Mirror: {endpoint}")
    print(f"  Target: {output_dir}")
    print()

    try:
        snapshot_download(
            repo_id=model_id,
            local_dir=str(output_dir),
            local_dir_use_symlinks=False,
            resume_download=resume,
            max_workers=4,
        )
        return True
    except Exception as e:
        print(f"  Download failed: {e}")
        return False


def _download_from_modelscope(output_dir: Path) -> bool:
    """Download using modelscope SDK. Returns True on success."""
    try:
        from modelscope import snapshot_download
    except ImportError:
        print("  [INFO] modelscope not installed. Trying: pip install modelscope")
        import subprocess
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "modelscope", "-q"]
        )
        from modelscope import snapshot_download

    print(f"  Downloading Qwen/Qwen2.5-7B from ModelScope ...")
    print(f"  Target: {output_dir}")
    print()

    try:
        snapshot_download(
            "Qwen/Qwen2.5-7B",
            cache_dir=str(output_dir.parent),
            local_dir=str(output_dir),
        )
        return True
    except Exception as e:
        print(f"  ModelScope download failed: {e}")
        return False


def _verify_model(model_dir: Path) -> bool:
    """Verify the model directory has all required files."""
    required = ["config.json", "tokenizer_config.json", "tokenizer.json"]
    for fname in required:
        if not (model_dir / fname).exists():
            print(f"  [WARN] Missing: {fname}")
            return False
    # Check for weight files
    weights = list(model_dir.glob("*.safetensors")) + list(model_dir.glob("*.bin"))
    if not weights:
        # Check index file for sharded weights
        idx = model_dir / "model.safetensors.index.json"
        if idx.exists():
            import json
            with open(idx) as f:
                index = json.load(f)
            shards = set(index.get("weight_map", {}).values())
            missing = [s for s in shards if not (model_dir / s).exists()]
            if missing:
                print(f"  [WARN] Missing {len(missing)} weight shards")
                return False
        else:
            print("  [WARN] No weight files found")
            return False
    return True


def ensure_base_model(
    model_dir: str | Path = None,
    model_id: str = "Qwen/Qwen2.5-7B",
    source: str = "auto",
    verbose: bool = True,
) -> Path:
    """Ensure the base model is available locally. Downloads if missing.

    Args:
        model_dir: Local directory for the model. Default: models/qwen2.5-7b
        model_id: HuggingFace model ID.
        source: "auto" (try mirrors then ModelScope), "huggingface", "modelscope"
        verbose: Print progress messages.

    Returns:
        Path to the local model directory.
    """
    if model_dir is None:
        model_dir = DEFAULT_MODEL_DIR
    model_dir = Path(model_dir)

    # Already exists and is valid
    if model_dir.is_dir() and _verify_model(model_dir):
        if verbose:
            total_gb = sum(
                f.stat().st_size for f in model_dir.rglob("*") if f.is_file()
            ) / 1e9
            print(f"Base model found: {model_dir} ({total_gb:.1f} GB)")
        return model_dir

    # Needs download
    model_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("=" * 60)
        print("  Base model not found. Downloading Qwen2.5-7B ...")
        print(f"  Target: {model_dir}")
        print("=" * 60)
        print()

    success = False

    if source == "modelscope":
        success = _download_from_modelscope(model_dir)
    elif source == "huggingface":
        os.environ["HF_ENDPOINT"] = "https://huggingface.co"
        success = _download_from_hf_hub(model_id, model_dir, "https://huggingface.co")
    else:  # auto
        # Try HF mirrors first
        for endpoint, name in HF_MIRRORS:
            if verbose:
                print(f"Trying {name} ({endpoint}) ...")
            if _test_endpoint(endpoint, timeout=3.0):
                success = _download_from_hf_hub(model_id, model_dir, endpoint)
                if success:
                    break
            elif verbose:
                print(f"  {name} unreachable, trying next ...")
            time.sleep(0.5)

        # Fallback to ModelScope
        if not success:
            if verbose:
                print("  HF mirrors failed. Trying ModelScope ...")
            success = _download_from_modelscope(model_dir)

    if not success:
        print()
        print("=" * 60)
        print("  ALL download sources failed.")
        print()
        print("  Manual download options:")
        print(f"  1. HF Mirror:  git clone https://hf-mirror.com/{model_id}")
        print(f"  2. ModelScope: git clone https://modelscope.cn/{model_id}.git")
        print(f"  3. Official HF: git clone https://huggingface.co/{model_id}")
        print()
        print(f"  Place the model files in: {model_dir}")
        print("=" * 60)
        sys.exit(1)

    # Verify
    if _verify_model(model_dir):
        total_gb = sum(
            f.stat().st_size for f in model_dir.rglob("*") if f.is_file()
        ) / 1e9
        if verbose:
            print(f"  Done! Model saved: {model_dir} ({total_gb:.1f} GB)")
        return model_dir
    else:
        print("  [ERROR] Download completed but verification failed.")
        print(f"  Check: {model_dir}")
        sys.exit(1)


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Ensure Qwen2.5-7B base model is available locally"
    )
    parser.add_argument("--output", default=str(DEFAULT_MODEL_DIR),
                        help="Target directory for the model")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B",
                        help="HuggingFace model ID")
    parser.add_argument("--source", default="auto",
                        choices=["auto", "huggingface", "modelscope"],
                        help="Download source (default: auto)")
    parser.add_argument("--force", action="store_true",
                        help="Re-download even if model exists")
    args = parser.parse_args()

    model_dir = Path(args.output)
    if args.force and model_dir.is_dir():
        import shutil
        print(f"Removing existing model: {model_dir}")
        shutil.rmtree(model_dir)

    path = ensure_base_model(
        model_dir=model_dir,
        model_id=args.model,
        source=args.source,
    )
    print(f"\nModel ready: {path}")


if __name__ == "__main__":
    main()
