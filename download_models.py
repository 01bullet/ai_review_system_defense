"""
Download / install the fine-tuned StruQ LoRA adapters for local review.

The base model (Qwen2.5-7B, ~15 GB) is downloaded separately by
`ensure_model.py`.  This script installs the *defended reviewer* adapters
(~2.2 GB each) into `models/`.  Without an adapter the local reviewer
falls back to the undefended base model.

Usage:
    python download_models.py                       # install v2a (default)
    python download_models.py --all                 # v2a + v3b
    python download_models.py --list                # show adapter info
    python download_models.py --local-archive PATH  # install from a local .tar.gz
    python download_models.py --local-dir PATH      # install from a local dir
    python download_models.py --adapter v2a --local-dir PATH

Sources (in priority order):
  1. Already installed in models/            -> skip
  2. --local-archive / --local-dir          -> copy / extract
  3. ADAPTERS[*]["urls"]                     -> download + extract

If the adapter is only available on your own machine / release attachment,
pass --local-archive or --local-dir.  To host it yourself, put a direct URL
in ADAPTERS below.
"""

import argparse
import os
import shutil
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT / "models"

# Adapter definitions.  "urls" are placeholders — replace with your own
# hosted mirrors (e.g. a release attachment / ModelScope / HuggingFace).
ADAPTERS = {
    "v2a": {
        "dest": "models/struq_v2_a/struq_lora_adapter",
        "version": "v2_a",
        "desc": "两阶段（人工对齐 + API 防御）— 唯一评估确认生产可用的审稿模型",
        "required": ["adapter_config.json", "adapter_model.safetensors", "struq_config.json"],
        "urls": [],
    },
    "v3b": {
        "dest": "models/struq_v3/v3b_api_defense",
        "version": "v3",
        "desc": "三阶段 PeerRead + API 扩展 — 重训版（max_seq_length=6144，已修复全 1 分崩溃）",
        "required": ["adapter_config.json", "adapter_model.safetensors", "struq_config.json"],
        "urls": [],
    },
}


def _installed(adapter) -> bool:
    dest = PROJECT / adapter["dest"]
    return all((dest / f).is_file() for f in adapter["required"])


def _is_archive(path: Path) -> bool:
    return path.suffix in (".tar.gz", ".tgz", ".zip")


def _extract_archive(archive: Path, workdir: Path) -> Path:
    """Extract archive into workdir, return the folder containing the adapter files."""
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as z:
            z.extractall(workdir)
    else:
        with tarfile.open(archive, "r:*") as t:
            t.extractall(workdir, filter="data")

    # Find the dir that contains the adapter files
    for f in workdir.rglob("adapter_config.json"):
        return f.parent
    # Fall back to the top of workdir
    return workdir


def _copy_dir(src: Path, dest: Path, adapter):
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dest, dirs_exist_ok=True)
    missing = [f for f in adapter["required"] if not (dest / f).is_file()]
    if missing:
        shutil.rmtree(dest, ignore_errors=True)
        raise SystemExit(f"[ERROR] {src} is not a valid adapter — missing: {missing}")


def _install_from_remote(adapter, name):
    if not adapter["urls"]:
        raise SystemExit(
            f"[ERROR] No remote URL configured for '{name}' and no local archive/dir given.\n"
            f"        Host the adapter (~2.2 GB) somewhere and add its URL to ADAPTERS "
            f"in download_models.py, or pass --local-archive / --local-dir."
        )
    dest = PROJECT / adapter["dest"]
    with tempfile.TemporaryDirectory() as tmp:
        for url in adapter["urls"]:
            print(f"  Downloading {name} from {url} ...")
            arch = Path(tmp) / f"{name}.tar.gz"
            try:
                urllib.request.urlretrieve(url, arch)
            except Exception as e:
                print(f"    failed: {e}")
                continue
            workdir = Path(tmp) / "x"
            src = _extract_archive(arch, workdir)
            _copy_dir(src, dest, adapter)
            print(f"  [OK] {name} installed to {adapter['dest']}")
            return
    raise SystemExit(f"[ERROR] All remote sources failed for '{name}'.")


def install(name, local_archive=None, local_dir=None):
    adapter = ADAPTERS[name]
    if _installed(adapter):
        print(f"[OK] {name} already installed at {adapter['dest']} — skipping.")
        return

    print(f"Installing {name}: {adapter['desc']}")
    dest = PROJECT / adapter["dest"]
    dest.parent.mkdir(parents=True, exist_ok=True)

    if local_dir:
        _copy_dir(Path(local_dir), dest, adapter)
        print(f"  [OK] {name} installed from local dir -> {adapter['dest']}")
        return

    if local_archive:
        arch = Path(local_archive)
        with tempfile.TemporaryDirectory() as tmp:
            src = _extract_archive(arch, Path(tmp))
            _copy_dir(src, dest, adapter)
        print(f"  [OK] {name} installed from {arch.name} -> {adapter['dest']}")
        return

    _install_from_remote(adapter, name)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="list adapters and status")
    ap.add_argument("--all", action="store_true", help="install all adapters")
    ap.add_argument("--adapter", choices=list(ADAPTERS), help="adapter to install")
    ap.add_argument("--local-archive", help="path to a local .tar.gz/.zip of the adapter")
    ap.add_argument("--local-dir", help="path to a local adapter directory")
    args = ap.parse_args()

    if args.list:
        for name, a in ADAPTERS.items():
            status = "installed" if _installed(a) else "missing"
            print(f"  {name:<5} {status:<9} {a['dest']}  —  {a['desc']}")
        return

    targets = [args.adapter] if args.adapter else (
        list(ADAPTERS) if args.all else ["v2a"]
    )
    for name in targets:
        install(name, local_archive=args.local_archive, local_dir=args.local_dir)

    print("\nBase model (Qwen2.5-7B) is installed separately via: python ensure_model.py")


if __name__ == "__main__":
    main()
