"""
Download the base model to a local directory for offline training.

Usage:
  python struq_defense/download_model.py                      # default: models/qwen2.5-7b
  python struq_defense/download_model.py --model Qwen/Qwen2.5-7B --output models/qwen2.5-7b
  python struq_defense/download_model.py --model meta-llama/Llama-3.1-8B  # needs HF token

After download, set environment variable or edit config.py:
  export STRUQ_LOCAL_MODEL=models/qwen2.5-7b
  # or in config.py: LOCAL_MODEL_PATH = "models/qwen2.5-7b"

The script always uses HF_ENDPOINT=https://hf-mirror.com for China mainland.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

# Always use mirror in China mainland
if "HF_ENDPOINT" not in os.environ:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


def download_model(
    model_id: str = "Qwen/Qwen2.5-7B",
    output_dir: str = "models/qwen2.5-7b",
    token: str | None = None,
    resume: bool = True,
    verbose: bool = True,
):
    """Download a HuggingFace model to local directory.

    Downloads all required files: config.json, tokenizer files,
    model weights (safetensors), and generation_config.json.

    Args:
        model_id: HuggingFace model ID (e.g. "Qwen/Qwen2.5-7B").
        output_dir: Local directory to save the model.
        token: HuggingFace token for gated models (e.g. Llama).
        resume: Resume partial downloads.
        verbose: Print progress.
    """
    from huggingface_hub import snapshot_download

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"Downloading {model_id} → {out}")
        print(f"  Mirror: {os.environ.get('HF_ENDPOINT', 'huggingface.co')}")
        if resume:
            print(f"  Resume: enabled (existing files will be skipped)")
        print()

    try:
        snapshot_download(
            repo_id=model_id,
            local_dir=str(out),
            local_dir_use_symlinks=False,
            resume_download=resume,
            token=token,
            max_workers=4,
        )
    except Exception as e:
        print(f"Error: {e}")
        print()
        print("Troubleshooting:")
        print("  1. Check network: curl -I https://hf-mirror.com")
        print("  2. Try other mirror: export HF_ENDPOINT=https://hf.xeduapi.com")
        print(f"  3. For gated models (Llama), request access at huggingface.co/{model_id}")
        print(f"     Then run: python download_model.py --model {model_id} --token hf_xxx")
        sys.exit(1)

    # Verify download
    required = ["config.json", "tokenizer_config.json", "tokenizer.json"]
    missing = [f for f in required if not (out / f).exists()]
    if missing:
        print(f"Warning: Missing files: {missing}")
    else:
        # Count safetensors / bin files
        weights = list(out.glob("*.safetensors")) + list(out.glob("*.bin"))
        total_gb = sum(f.stat().st_size for f in out.rglob("*")) / 1e9
        print(f"  Weights: {len(weights)} files, {total_gb:.1f} GB total")
        print(f"  Done! Model saved to {out}")

    # Print instructions
    print()
    print("=" * 60)
    print("  Next steps:")
    print("=" * 60)
    print(f"  1. Set env var: export STRUQ_LOCAL_MODEL={out}")
    print(f"  2. Or edit struq_defense/config.py: LOCAL_MODEL_PATH = '{out}'")
    print(f"  3. Test: python -c \"from transformers import AutoTokenizer;")
    print(f"            t = AutoTokenizer.from_pretrained('{out}'); print('OK')\"")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Download HuggingFace model for offline training"
    )
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B",
                        help="HuggingFace model ID")
    parser.add_argument("--output", default="models/qwen2.5-7b",
                        help="Output directory")
    parser.add_argument("--token", default=None,
                        help="HF token for gated models")
    parser.add_argument("--no-resume", action="store_true",
                        help="Do not resume partial downloads")
    args = parser.parse_args()

    download_model(
        model_id=args.model,
        output_dir=args.output,
        token=args.token,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()
