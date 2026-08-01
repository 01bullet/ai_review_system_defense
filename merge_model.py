"""
Merge LoRA adapter with base model locally.

After downloading the LoRA adapter from AutoDL, run this script
to produce a standalone merged model (no PEFT needed for inference).

Usage:
  python merge_model.py
  python merge_model.py --adapter ./models/struq_lora_adapter --base ./models/qwen2.5-7b --output ./models/struq_merged
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))

if "HF_ENDPOINT" not in os.environ:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


def merge(
    adapter_path: str,
    base_model: str,
    output_dir: str,
    verbose: bool = True,
):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel

    if verbose:
        print(f"Base model: {base_model}")
        print(f"Adapter:   {adapter_path}")
        print(f"Output:    {output_dir}")
        print()

    # 1) Load tokenizer from adapter (has special tokens from training)
    if verbose:
        print("Loading tokenizer from adapter...")
    tokenizer = AutoTokenizer.from_pretrained(
        adapter_path,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"  Tokenizer vocab: {len(tokenizer)}")

    # 2) Load base model
    if verbose:
        print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )
    print(f"  Model vocab: {model.config.vocab_size}")

    # 3) Resize embeddings to match tokenizer (add special tokens if needed)
    if len(tokenizer) != model.config.vocab_size:
        print(f"  Resizing embeddings: {model.config.vocab_size} -> {len(tokenizer)}")
        model.resize_token_embeddings(len(tokenizer))

        # Initialize special token embeddings from similar tokens
        struq_config_path = os.path.join(adapter_path, "struq_config.json")
        if os.path.exists(struq_config_path):
            with open(struq_config_path, "r") as f:
                struq_cfg = json.load(f)
            special_tokens = struq_cfg.get("special_tokens", [])
            if special_tokens:
                from struq_defense.frontend import SecureFrontend
                frontend = SecureFrontend()
                frontend.initialize_special_embeddings(model, tokenizer)
                print(f"  Initialized {len(special_tokens)} special token embeddings")

    # 4) Load and merge LoRA adapter
    if verbose:
        print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(model, adapter_path)

    if verbose:
        print("Merging weights...")
    merged = model.merge_and_unload()

    # 5) Strip quantization_config — merged model is full bf16, not quantized
    if hasattr(merged, 'config') and hasattr(merged.config, 'quantization_config'):
        del merged.config.quantization_config

    # 6) Save
    os.makedirs(output_dir, exist_ok=True)
    if verbose:
        print(f"Saving merged model to {output_dir}...")
    merged.save_pretrained(output_dir, safe_serialization=True,
                           max_shard_size="4GB")
    tokenizer.save_pretrained(output_dir)

    if verbose:
        total_gb = sum(f.stat().st_size for f in Path(output_dir).rglob("*")) / 1e9
        print(f"Merged model saved: {total_gb:.1f} GB")
        print("Done!")


def main():
    parser = argparse.ArgumentParser(
        description="Merge LoRA adapter with base model"
    )
    parser.add_argument("--adapter", type=str,
                        default="models/struq/struq_lora_adapter",
                        help="Path to LoRA adapter directory")
    parser.add_argument("--base", type=str,
                        default="models/qwen2.5-7b",
                        help="Path to base model directory")
    parser.add_argument("--output", type=str,
                        default="models/struq/struq_merged_model",
                        help="Output directory for merged model")
    args = parser.parse_args()

    merge(
        adapter_path=args.adapter,
        base_model=args.base,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
