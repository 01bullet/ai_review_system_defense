"""
Unified CLI for the StruQ defense system.

Subcommands:
  build-dataset  — Build structured instruction tuning dataset
  train          — Run structured instruction tuning (QLoRA)
  review         — Review a paper using the defended local model
  evaluate       — Evaluate defense against various attacks
  info           — Print system status and configuration

Usage:
  python -m struq_defense.run build-dataset --papers example_papers/
  python -m struq_defense.run train --dataset data/struq_dataset.json
  python -m struq_defense.run review --paper example_papers/xxx/xxx.pdf
  python -m struq_defense.run evaluate --papers example_papers/
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Set HF mirror before any transformers imports (for China mainland)
if "HF_ENDPOINT" not in os.environ:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

# Redirect HF cache to D: drive if available (C: drive too small for models)
_D_DRIVE_CACHE = "D:/huggingface_cache"
if os.path.exists("D:/") and "HF_HOME" not in os.environ:
    os.environ["HF_HOME"] = _D_DRIVE_CACHE
    os.makedirs(_D_DRIVE_CACHE, exist_ok=True)


def _detect_base_model(adapter_path: str | None, cli_override: str | None = None) -> str:
    """Detect the correct base model for an adapter.

    Checks struq_config.json for version field:
      - "v2" → Qwen/Qwen2.5-7B-Instruct
      - "v2_a" → Qwen/Qwen2.5-7B (base model)
      - otherwise → Qwen/Qwen2.5-7B
    CLI --base-model overrides auto-detection.
    """
    if cli_override:
        return cli_override
    if adapter_path:
        cfg_file = os.path.join(adapter_path, "struq_config.json")
        if os.path.exists(cfg_file):
            try:
                with open(cfg_file, "r") as f:
                    cfg = json.load(f)
                version = cfg.get("version", "")
                if version == "v2":
                    from struq_defense.config_v2 import BASE_MODEL as V2_BASE
                    return V2_BASE
                if version == "v2_a":
                    from struq_defense.config_v2_a import BASE_MODEL as V2A_BASE
                    return V2A_BASE
            except Exception:
                pass
    from struq_defense.config import BASE_MODEL
    return BASE_MODEL


def cmd_build_dataset(args):
    """Build the structured instruction tuning dataset."""
    from struq_defense.dataset import StruqDatasetBuilder
    from struq_defense.config import DATASET_OUTPUT, EXAMPLE_PAPERS_DIR

    print("=" * 60)
    print("  StruQ Dataset Builder")
    print("=" * 60)

    papers_dir = Path(args.papers) if args.papers else EXAMPLE_PAPERS_DIR
    if not papers_dir.is_absolute():
        papers_dir = (PROJECT / args.papers).resolve()

    print(f"  Papers directory: {papers_dir}")
    print(f"  Output: {args.output or DATASET_OUTPUT}")
    print(f"  Clean ratio: {args.clean_ratio}")
    print(f"  Naive variants: {args.naive_variants}")
    print(f"  Completion variants: {args.completion_variants}")
    print(f"  Skip API reviews: {args.skip_reviews}")
    print()

    builder = StruqDatasetBuilder()
    dataset = builder.build(
        papers_dir=papers_dir,
        max_papers=args.max_papers,
        skip_review_generation=args.skip_reviews,
        verbose=True,
        naive_variants=args.naive_variants,
        completion_variants=args.completion_variants,
        clean_ratio=args.clean_ratio,
    )

    if args.with_rules:
        extra = builder.build_extra_with_rule_strategies(
            builder.discover_papers(papers_dir)[:args.max_papers],
        )
        dataset.extend(extra)
        print(f"  Added {len(extra)} rule-based attack samples")

    builder.save(dataset, args.output)
    print("\nDone! Next step: python -m struq_defense.run train")


def cmd_train(args):
    """Run structured instruction tuning."""
    from struq_defense.train import train_structured_instruction_tuning
    from struq_defense.config import LORA_OUTPUT, DATASET_OUTPUT

    print("=" * 60)
    print("  StruQ Structured Instruction Tuning")
    print("=" * 60)

    dataset_path = args.dataset or DATASET_OUTPUT
    output_dir = args.output or LORA_OUTPUT

    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found at {dataset_path}")
        print("Run 'python -m struq_defense.run build-dataset' first.")
        sys.exit(1)

    print(f"  Base model: {args.model}")
    print(f"  Dataset: {dataset_path}")
    print(f"  Output: {output_dir}")
    print(f"  Epochs: {args.epochs}")
    print(f"  LR: {args.lr}")
    if args.merge:
        print(f"  Merge: {args.merge_output}")
    print()

    kwargs = {}
    if args.model is not None:
        kwargs["base_model"] = args.model
    if args.epochs is not None:
        kwargs["num_epochs"] = args.epochs
    if args.lr is not None:
        kwargs["learning_rate"] = args.lr

    result = train_structured_instruction_tuning(
        dataset_path=dataset_path,
        output_dir=output_dir,
        resume_from_checkpoint=not args.no_resume,
        verbose=True,
        **kwargs,
    )

    if args.merge:
        from struq_defense.train import merge_and_save
        merge_and_save(
            result["model"],
            result["tokenizer"],
            output_dir=args.merge_output,
        )
        print(f"Merged model saved to {args.merge_output}")

    print("\nDone! Next step: python -m struq_defense.run review --paper <path>")


def cmd_review(args):
    """Review a paper using the defended local model."""
    from struq_defense.reviewer import LocalLLMReviewer
    from ai_scientist.gan_defense.data_utils import extract_text_from_latex_fast

    print("=" * 60)
    print("  StruQ Local LLM Reviewer")
    print("=" * 60)

    # Resolve paper path
    paper_path = Path(args.paper)
    if not paper_path.is_absolute():
        paper_path = (PROJECT / args.paper).resolve()

    if not paper_path.exists():
        print(f"Error: Path not found: {paper_path}")
        sys.exit(1)

    if paper_path.is_file() and paper_path.suffix.lower() == ".pdf":
        paper_dir = paper_path.parent
        paper_name = paper_path.stem
    elif paper_path.is_dir():
        paper_dir = paper_path
        paper_name = paper_dir.name
    else:
        print(f"Error: Unrecognized path: {paper_path}")
        sys.exit(1)

    # Read paper
    tex_file = paper_dir / "latex" / "template.tex"
    if not tex_file.exists():
        print(f"Error: No latex/template.tex in {paper_dir}")
        sys.exit(1)

    latex_content = tex_file.read_text(encoding="utf-8")
    paper_text = extract_text_from_latex_fast(latex_content)

    print(f"  Paper: {paper_name}")
    print(f"  Text length: {len(paper_text):,} chars")

    # Load model — auto-detect base model from adapter version
    base_model = _detect_base_model(args.adapter, args.base_model)
    reviewer = LocalLLMReviewer()
    if args.merged:
        reviewer.load(merged_path=args.merged, use_merged=True)
    else:
        reviewer.load(adapter_path=args.adapter, base_model=base_model)

    # Review
    print("  Reviewing...")
    result = reviewer.review(paper_text, temperature=args.temperature)

    # Output
    print()
    print("-" * 40)
    print("  Review Result:")
    print("-" * 40)
    for key in ["novelty", "soundness", "presentation", "overall", "decision"]:
        print(f"  {key.capitalize():12s}: {result.get(key, 'N/A')}")
    review_text = result.get("review", "")
    if review_text:
        print(f"\n  Review text:")
        print(f"  {review_text[:500]}...")

    if result.get("_defense"):
        print(f"\n  Defense: {result['_defense']['method']} ({result['_defense']['model']})")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\n  Saved to: {args.output}")


def cmd_evaluate(args):
    """Evaluate the defense against attacks."""
    import random
    from struq_defense.reviewer import LocalLLMReviewer
    from struq_defense.dataset import StruqDatasetBuilder
    from struq_defense.config import EXAMPLE_PAPERS_DIR

    print("=" * 60)
    print("  StruQ Defense Evaluation")
    print("=" * 60)

    # Load model — auto-detect base model from adapter version
    base_model = _detect_base_model(args.adapter, args.base_model)
    reviewer = LocalLLMReviewer()
    if args.merged:
        reviewer.load(merged_path=args.merged, use_merged=True)
    else:
        reviewer.load(adapter_path=args.adapter, base_model=base_model)

    # Discover papers
    builder = StruqDatasetBuilder()
    papers = builder.discover_papers(Path(args.papers))
    if args.max_papers:
        random.shuffle(papers)
        papers = papers[:args.max_papers]

    if not papers:
        print("No papers found.")
        sys.exit(1)

    print(f"  Testing on {len(papers)} papers")
    print()

    results = {
        "clean": [],
        "naive_attack": [],
        "completion_attack": [],
    }

    for paper_dir in papers:
        paper_name = paper_dir.name
        latex_content, clean_text = builder.read_paper_text(paper_dir)

        # Test 1: Clean paper
        clean_result = reviewer.review(clean_text)
        results["clean"].append({
            "paper": paper_name,
            "overall": clean_result.get("overall", 0),
            "decision": clean_result.get("decision", "?"),
        })

        # Test 2: Naive attack
        try:
            attacked_text = builder.generate_naive_attack(latex_content, clean_text)
            naive_result = reviewer.review(attacked_text)
            results["naive_attack"].append({
                "paper": paper_name,
                "overall": naive_result.get("overall", 0),
                "decision": naive_result.get("decision", "?"),
                "delta_from_clean": naive_result.get("overall", 0) - clean_result.get("overall", 0),
            })
        except Exception as e:
            print(f"  Naive attack failed for {paper_name}: {e}")

        # Test 3: Completion attack
        try:
            attacked_text = builder.generate_completion_attack(latex_content, clean_text)
            compl_result = reviewer.review(attacked_text)
            results["completion_attack"].append({
                "paper": paper_name,
                "overall": compl_result.get("overall", 0),
                "decision": compl_result.get("decision", "?"),
                "delta_from_clean": compl_result.get("overall", 0) - clean_result.get("overall", 0),
            })
        except Exception as e:
            print(f"  Completion attack failed for {paper_name}: {e}")

        print(f"  {paper_name}: clean={clean_result.get('overall',0)}, "
              f"naive_delta={results['naive_attack'][-1].get('delta_from_clean','N/A') if results['naive_attack'] else 'N/A'}, "
              f"compl_delta={results['completion_attack'][-1].get('delta_from_clean','N/A') if results['completion_attack'] else 'N/A'}")

    # Summary
    print()
    print("=" * 60)
    print("  Evaluation Summary")
    print("=" * 60)

    for attack_type, items in results.items():
        if not items:
            continue
        avg_score = sum(r.get("overall", 0) for r in items) / len(items)
        if attack_type != "clean":
            avg_delta = sum(r.get("delta_from_clean", 0) for r in items) / len(items)
            print(f"  {attack_type:20s}: avg score={avg_score:.1f}, avg delta={avg_delta:+.1f}")
        else:
            print(f"  {attack_type:20s}: avg score={avg_score:.1f}")
        decisions = [r.get("decision", "?") for r in items]
        accepts = sum(1 for d in decisions if "Accept" in str(d))
        print(f"  {'':20s}  decisions: {accepts} Accept / {len(items) - accepts} Reject")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\nResults saved to {args.output}")


def cmd_info(args):
    """Print system status and configuration."""
    from struq_defense.config import (
        BASE_MODEL, LOCAL_MODEL_PATH, SPECIAL_TOKENS, FILTER_STRINGS, EMBED_INIT_MAP,
        CLEAN_RATIO, NAIVE_ATTACK_RATIO, COMPLETION_ATTACK_RATIO,
        LORA_R, LORA_ALPHA, LORA_OUTPUT, DATASET_OUTPUT,
        NUM_EPOCHS, LEARNING_RATE, MAX_SEQ_LENGTH, DEVICE,
        EXAMPLE_PAPERS_DIR, SPECIAL_TOKENS as _st,
    )

    print("=" * 60)
    print("  StruQ Defense — System Info")
    print("=" * 60)

    print(f"\n  Device: {DEVICE}")
    print(f"  Base model: {BASE_MODEL}")
    local_path = LOCAL_MODEL_PATH or os.environ.get("STRUQ_LOCAL_MODEL", "")
    if local_path and os.path.isdir(local_path):
        size = sum(f.stat().st_size for f in Path(local_path).rglob("*")) / 1e9
        print(f"  Local model: {local_path} ({size:.1f} GB) ✓")
    elif local_path:
        print(f"  Local model: {local_path} (NOT FOUND)")
    else:
        print(f"  Local model: not set (will download from HF Hub)")
    print(f"  Project root: {PROJECT}")

    print(f"\n  [Special Tokens]")
    print(f"  {SPECIAL_TOKENS}")
    print(f"\n  [Filter Strings]")
    print(f"  {FILTER_STRINGS}")
    print(f"\n  [Embed Init Map]")
    for tok, src in EMBED_INIT_MAP.items():
        print(f"  {tok} ← {src}")

    print(f"\n  [Dataset Ratios]")
    print(f"  Clean: {CLEAN_RATIO:.0%}")
    print(f"  Naive: {NAIVE_ATTACK_RATIO:.0%}")
    print(f"  Completion: {COMPLETION_ATTACK_RATIO:.0%}")

    print(f"\n  [Training]")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  LR: {LEARNING_RATE}")
    print(f"  LoRA r={LORA_R}, alpha={LORA_ALPHA}")
    print(f"  Max seq length: {MAX_SEQ_LENGTH}")

    print(f"\n  [Paths]")
    print(f"  Dataset: {DATASET_OUTPUT}")
    print(f"  LoRA output: {LORA_OUTPUT}")
    print(f"  Example papers: {EXAMPLE_PAPERS_DIR}")

    # Check dataset status
    if os.path.exists(DATASET_OUTPUT):
        try:
            with open(DATASET_OUTPUT, "r") as f:
                ds = json.load(f)
            print(f"\n  [Dataset Status]")
            print(f"  File: {DATASET_OUTPUT} — {len(ds)} entries")
            types = {}
            for d in ds:
                types[d["type"]] = types.get(d["type"], 0) + 1
            for t, c in sorted(types.items()):
                print(f"    {t}: {c}")
        except:
            print(f"  Dataset exists but could not be read")
    else:
        print(f"\n  [Dataset Status]")
        print(f"  Not yet built. Run: python -m struq_defense.run build-dataset")

    # Check model status
    if os.path.exists(LORA_OUTPUT):
        print(f"\n  [Model Status]")
        print(f"  LoRA adapter: {LORA_OUTPUT} — exists")
    else:
        print(f"\n  [Model Status]")
        print(f"  Not yet trained. Run: python -m struq_defense.run train")


# ---- V2_A commands ----

def cmd_v2a_stage1(args):
    """Run V2_A Stage 1 format training."""
    from struq_defense.train_v2_a import build_stage1_dataset, train_stage1_format
    from struq_defense.config_v2_a import DATASET_STAGE1_OUTPUT, STAGE1_OUTPUT

    print("=" * 60)
    print("  V2_A Stage 1: Format Training")
    print("=" * 60)

    # Build or load dataset
    dataset = None
    if args.dataset:
        print(f"Loading dataset: {args.dataset}")
        with open(args.dataset, "r", encoding="utf-8") as f:
            dataset = json.load(f)
    else:
        papers_dir = Path(args.papers) if args.papers else None
        if papers_dir and not papers_dir.is_absolute():
            papers_dir = (PROJECT / args.papers).resolve()
        print("Building Stage 1 dataset...")
        dataset = build_stage1_dataset(papers_dir=papers_dir, max_papers=args.max_papers)
        os.makedirs(os.path.dirname(DATASET_STAGE1_OUTPUT), exist_ok=True)
        with open(DATASET_STAGE1_OUTPUT, "w", encoding="utf-8") as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
        print(f"Saved to {DATASET_STAGE1_OUTPUT}")

    # Resolve model path
    model_path = args.model
    if not model_path:
        from struq_defense.config_v2_a import LOCAL_MODEL_PATH, BASE_MODEL
        if LOCAL_MODEL_PATH and os.path.isdir(LOCAL_MODEL_PATH):
            model_path = LOCAL_MODEL_PATH
        else:
            model_path = BASE_MODEL

    output = args.output or STAGE1_OUTPUT

    kwargs = {}
    if args.epochs is not None:
        kwargs["num_epochs"] = args.epochs
    if args.lr is not None:
        kwargs["learning_rate"] = args.lr

    train_stage1_format(
        dataset=dataset,
        output_dir=output,
        base_model=model_path,
        resume_from_checkpoint=not args.no_resume,
        **kwargs,
    )

    print(f"\nStage 1 complete! Adapter saved to: {output}")
    print(f"Next: python -m struq_defense.run v2a-stage2")


def cmd_v2a_stage2(args):
    """Run V2_A Stage 2 defense training."""
    from struq_defense.train_v2_a import build_stage2_dataset, train_stage2_defense
    from struq_defense.config_v2_a import (
        DATASET_STAGE2_OUTPUT, STAGE1_OUTPUT, STAGE2_OUTPUT,
        LOCAL_MODEL_PATH, BASE_MODEL,
    )

    print("=" * 60)
    print("  V2_A Stage 2: Defense Training")
    print("=" * 60)

    # Build or load dataset
    dataset = None
    if args.dataset:
        print(f"Loading dataset: {args.dataset}")
        with open(args.dataset, "r", encoding="utf-8") as f:
            dataset = json.load(f)
    else:
        papers_dir = Path(args.papers) if args.papers else None
        if papers_dir and not papers_dir.is_absolute():
            papers_dir = (PROJECT / args.papers).resolve()
        print("Building Stage 2 dataset...")
        dataset = build_stage2_dataset(
            papers_dir=papers_dir,
            max_papers=args.max_papers,
            skip_review_generation=args.skip_reviews,
        )
        os.makedirs(os.path.dirname(DATASET_STAGE2_OUTPUT), exist_ok=True)
        with open(DATASET_STAGE2_OUTPUT, "w", encoding="utf-8") as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
        print(f"Saved to {DATASET_STAGE2_OUTPUT}")

    # Resolve model path
    model_path = args.model
    if not model_path:
        if LOCAL_MODEL_PATH and os.path.isdir(LOCAL_MODEL_PATH):
            model_path = LOCAL_MODEL_PATH
        else:
            model_path = BASE_MODEL

    stage1_ckpt = args.stage1_checkpoint or STAGE1_OUTPUT
    if not os.path.exists(os.path.join(stage1_ckpt, "adapter_model.safetensors")):
        print(f"Error: Stage 1 checkpoint not found at {stage1_ckpt}")
        print("Run 'python -m struq_defense.run v2a-stage1' first.")
        sys.exit(1)

    output = args.output or STAGE2_OUTPUT

    kwargs = {}
    if args.epochs is not None:
        kwargs["num_epochs"] = args.epochs
    if args.lr is not None:
        kwargs["learning_rate"] = args.lr

    train_stage2_defense(
        dataset=dataset,
        stage1_checkpoint=stage1_ckpt,
        output_dir=output,
        base_model=model_path,
        resume_from_checkpoint=not args.no_resume,
        use_label_masking=not args.no_label_masking,
        **kwargs,
    )

    print(f"\nStage 2 complete! Adapter saved to: {output}")
    print(f"Test: python -m struq_defense.run review "
          f"--paper example_papers/adaptive_dual_scale_denoising "
          f"--adapter {output}")


def cmd_v2a_pipeline(args):
    """Run full V2_A pipeline (Stage 1 + Stage 2)."""
    from struq_defense.train_v2_a import train_v2_a_full_pipeline

    papers_dir = Path(args.papers) if args.papers else None
    if papers_dir and not papers_dir.is_absolute():
        papers_dir = (PROJECT / args.papers).resolve()

    train_v2_a_full_pipeline(
        papers_dir=papers_dir,
        max_papers=args.max_papers,
        skip_review_generation=args.skip_reviews,
    )


# ---- CLI setup ----

def main():
    parser = argparse.ArgumentParser(
        description="StruQ Defense — Structured Query Defense for AI Scientist",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Subcommands")

    # ---- build-dataset ----
    p_build = subparsers.add_parser("build-dataset", help="Build training dataset")
    p_build.add_argument("--papers", type=str, default=None,
                         help="Papers directory (default: example_papers/)")
    p_build.add_argument("--output", type=str, default=None,
                         help="Output JSON path")
    p_build.add_argument("--max-papers", type=int, default=None,
                         help="Limit number of papers")
    p_build.add_argument("--clean-ratio", type=float, default=None,
                         help="Override clean sample ratio")
    p_build.add_argument("--naive-variants", type=int, default=None,
                         help="Naive attack variants per paper")
    p_build.add_argument("--completion-variants", type=int, default=None,
                         help="Completion attack variants per paper")
    p_build.add_argument("--skip-reviews", action="store_true",
                         help="Skip API review generation (placeholder)")
    p_build.add_argument("--with-rules", action="store_true",
                         help="Include rule-based attack samples")

    # ---- train ----
    p_train = subparsers.add_parser("train", help="Run structured instruction tuning")
    p_train.add_argument("--dataset", type=str, default=None,
                         help="Dataset JSON path")
    p_train.add_argument("--model", type=str, default=None,
                         help="Base model ID (default from config)")
    p_train.add_argument("--output", type=str, default=None,
                         help="LoRA adapter output dir")
    p_train.add_argument("--epochs", type=int, default=None,
                         help="Number of training epochs")
    p_train.add_argument("--lr", type=float, default=None,
                         help="Learning rate")
    p_train.add_argument("--merge", action="store_true",
                         help="Merge and save full model after training")
    p_train.add_argument("--merge-output", type=str, default=None,
                         help="Merged model output path")
    p_train.add_argument("--no-resume", action="store_true",
                         help="Do not resume from checkpoint")

    # ---- review ----
    p_review = subparsers.add_parser("review", help="Review a paper with defended model")
    p_review.add_argument("--paper", type=str, required=True,
                          help="Paper PDF path or directory")
    p_review.add_argument("--adapter", type=str, default=None,
                          help="LoRA adapter path")
    p_review.add_argument("--merged", type=str, default=None,
                          help="Merged model path (use merged model)")
    p_review.add_argument("--temperature", type=float, default=0.1,
                          help="Generation temperature")
    p_review.add_argument("--base-model", type=str, default=None,
                          help="Override base model (auto-detected from adapter)")
    p_review.add_argument("--output", type=str, default=None,
                          help="Save review to JSON file")

    # ---- evaluate ----
    p_eval = subparsers.add_parser("evaluate", help="Evaluate defense against attacks")
    p_eval.add_argument("--papers", type=str, default=None,
                        help="Papers directory")
    p_eval.add_argument("--max-papers", type=int, default=5,
                        help="Max papers to test")
    p_eval.add_argument("--adapter", type=str, default=None,
                        help="LoRA adapter path")
    p_eval.add_argument("--merged", type=str, default=None,
                        help="Merged model path")
    p_eval.add_argument("--base-model", type=str, default=None,
                        help="Override base model (auto-detected from adapter)")
    p_eval.add_argument("--output", type=str, default=None,
                        help="Save results to JSON")

    # ---- info ----
    p_info = subparsers.add_parser("info", help="Print system info and status")

    # ---- v2a-stage1 ----
    p_v2a_s1 = subparsers.add_parser("v2a-stage1", help="V2_A Stage 1: Format training")
    p_v2a_s1.add_argument("--dataset", type=str, default=None,
                          help="Dataset JSON path (auto-built if omitted)")
    p_v2a_s1.add_argument("--papers", type=str, default=None,
                          help="Papers directory")
    p_v2a_s1.add_argument("--max-papers", type=int, default=None)
    p_v2a_s1.add_argument("--output", type=str, default=None)
    p_v2a_s1.add_argument("--model", type=str, default=None,
                          help="Base model path override")
    p_v2a_s1.add_argument("--epochs", type=int, default=None)
    p_v2a_s1.add_argument("--lr", type=float, default=None)
    p_v2a_s1.add_argument("--no-resume", action="store_true")

    # ---- v2a-stage2 ----
    p_v2a_s2 = subparsers.add_parser("v2a-stage2", help="V2_A Stage 2: Defense training")
    p_v2a_s2.add_argument("--dataset", type=str, default=None,
                          help="Dataset JSON path (auto-built if omitted)")
    p_v2a_s2.add_argument("--papers", type=str, default=None)
    p_v2a_s2.add_argument("--max-papers", type=int, default=None)
    p_v2a_s2.add_argument("--stage1-checkpoint", type=str, default=None,
                          help="Path to Stage 1 adapter")
    p_v2a_s2.add_argument("--output", type=str, default=None)
    p_v2a_s2.add_argument("--model", type=str, default=None)
    p_v2a_s2.add_argument("--epochs", type=int, default=None)
    p_v2a_s2.add_argument("--lr", type=float, default=None)
    p_v2a_s2.add_argument("--no-resume", action="store_true")
    p_v2a_s2.add_argument("--no-label-masking", action="store_true")
    p_v2a_s2.add_argument("--skip-reviews", action="store_true")

    # ---- v2a-pipeline ----
    p_v2a_pipe = subparsers.add_parser("v2a-pipeline", help="V2_A full pipeline (Stage 1 + Stage 2)")
    p_v2a_pipe.add_argument("--papers", type=str, default=None)
    p_v2a_pipe.add_argument("--max-papers", type=int, default=None)
    p_v2a_pipe.add_argument("--skip-reviews", action="store_true")

    args = parser.parse_args()

    if args.command == "build-dataset":
        cmd_build_dataset(args)
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "review":
        cmd_review(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)
    elif args.command == "info":
        cmd_info(args)
    elif args.command == "v2a-stage1":
        cmd_v2a_stage1(args)
    elif args.command == "v2a-stage2":
        cmd_v2a_stage2(args)
    elif args.command == "v2a-pipeline":
        cmd_v2a_pipeline(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
