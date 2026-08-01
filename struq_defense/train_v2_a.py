"""
V2_A Two-Stage Training: Format training + Defense training on base model.

Stage 1 — Format Training:
  Teaches the base model to output proper JSON review format.
  Uses diverse review variants per paper (no attacks).

Stage 2 — Defense Training:
  Continues from Stage 1 checkpoint with full V2 defense dataset.
  Adds label masking to focus loss on [RESP] portion.
"""

from __future__ import annotations

import gc
import json
import os
import random
import sys
from pathlib import Path
from typing import Optional, Dict, List, Tuple

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

# HF mirror
if "HF_ENDPOINT" not in os.environ:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
from torch.utils.data import Dataset

from struq_defense.config_v2_a import (
    # Paths
    PROJECT as _P, DATA_DIR, MODELS_DIR,
    DATASET_STAGE1_OUTPUT, DATASET_STAGE2_OUTPUT,
    STAGE1_OUTPUT, STAGE2_OUTPUT,
    EXAMPLE_PAPERS_DIR,
    # Special tokens
    SPECIAL_TOKENS, EMBED_INIT_MAP, QUERY_TEMPLATE, REVIEW_PROMPT,
    FILTER_STRINGS, NEGATIVE_REVIEW_TARGET,
    # Stage 1 config
    STAGE1_NUM_EPOCHS, STAGE1_LEARNING_RATE,
    STAGE1_PER_DEVICE_BATCH_SIZE, STAGE1_GRADIENT_ACCUMULATION_STEPS,
    STAGE1_MAX_SEQ_LENGTH, STAGE1_LORA_R, STAGE1_LORA_ALPHA,
    STAGE1_LORA_TARGET_MODULES, STAGE1_WARMUP_RATIO, STAGE1_SAVE_STEPS,
    STAGE1_CLEAN_COPIES_PER_PAPER, STAGE1_REVIEW_VARIANTS,
    # Stage 2 config
    STAGE2_NUM_EPOCHS, STAGE2_LEARNING_RATE,
    STAGE2_PER_DEVICE_BATCH_SIZE, STAGE2_GRADIENT_ACCUMULATION_STEPS,
    STAGE2_MAX_SEQ_LENGTH, STAGE2_LORA_R, STAGE2_LORA_ALPHA,
    STAGE2_LORA_TARGET_MODULES, STAGE2_WARMUP_RATIO, STAGE2_SAVE_STEPS,
    STAGE2_EVAL_STEPS, STAGE2_VAL_SPLIT,
    STAGE2_CLEAN_COPIES_PER_PAPER, STAGE2_NAIVE_VARIANTS_PER_PAPER,
    STAGE2_COMPLETION_VARIANTS_PER_PAPER, STAGE2_FORMAT_VARIANTS_PER_PAPER,
    STAGE2_BOUNDARY_VARIANTS_PER_PAPER, STAGE2_NEGATIVE_TARGET_RATIO,
    # Attack configs
    FAKE_DELIMITERS, FAKE_RESPONSES, FAKE_RESPONSE_NOISE,
    FORMAT_ATTACKS, BOUNDARY_ATTACK_TEMPLATES,
    # General
    LORA_DROPOUT, WEIGHT_DECAY, MAX_GRAD_NORM, LOGGING_STEPS,
    LOAD_IN_4BIT, BNB_4BIT_COMPUTE_DTYPE, BNB_4BIT_QUANT_TYPE, BNB_4BIT_USE_DOUBLE_QUANT,
    BASE_MODEL, LOCAL_MODEL_PATH, DEVICE,
)

from struq_defense.frontend import SecureFrontend


# ============================================================
# Dataset wrapper (same as train.py)
# ============================================================

class StruqDatasetV2A(Dataset):
    """PyTorch Dataset for V2_A training."""

    def __init__(self, data: list[dict], tokenizer, max_length: int = 1024):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        text = self.data[idx]["text"]
        encoded = self.tokenizer(
            text, truncation=True, max_length=self.max_length,
            padding="max_length", return_tensors="pt",
        )
        input_ids = encoded["input_ids"][0]
        attention_mask = encoded["attention_mask"][0]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone(),
        }


def _collate_fn(batch):
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
    }


# ============================================================
# Label Masking Collate (V2)
# ============================================================

def create_label_masking_collate(tokenizer):
    """Mask labels before [MARK][RESP][COLN] — loss only on response."""
    resp_marker = "[MARK][RESP][COLN]"
    resp_marker_ids = tokenizer.encode(resp_marker, add_special_tokens=False)

    def _collate(batch):
        input_ids = torch.stack([b["input_ids"] for b in batch])
        attention_mask = torch.stack([b["attention_mask"] for b in batch])
        labels = input_ids.clone()

        for i in range(len(batch)):
            ids = input_ids[i].tolist()
            resp_pos = -1
            for j in range(len(ids) - len(resp_marker_ids) + 1):
                if ids[j:j + len(resp_marker_ids)] == resp_marker_ids:
                    resp_pos = j + len(resp_marker_ids)
                    break
            if resp_pos > 0:
                labels[i, :resp_pos] = -100
            else:
                labels[i, :] = -100
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    return _collate


# ============================================================
# Token initialization utils
# ============================================================

def _add_special_tokens(tokenizer, model):
    num_added = tokenizer.add_tokens(SPECIAL_TOKENS)
    if num_added > 0:
        model.resize_token_embeddings(len(tokenizer))
        print(f"  Added {num_added} special tokens, vocab size: {len(tokenizer)}")
    return num_added


def _initialize_special_embeddings(model, tokenizer):
    frontend = SecureFrontend()
    frontend.initialize_special_embeddings(model, tokenizer)
    for special_tok, source_text in frontend.embed_init_map.items():
        tok_id = tokenizer.convert_tokens_to_ids(special_tok)
        if tok_id != tokenizer.unk_token_id:
            source_id = tokenizer.encode(source_text, add_special_tokens=False)[0]
            cos = torch.cosine_similarity(
                model.get_input_embeddings().weight[tok_id].unsqueeze(0),
                model.get_input_embeddings().weight[source_id].unsqueeze(0),
            )
            print(f"  Embed init: {special_tok} <- {source_text} (cos_sim={cos.item():.4f})")


# ============================================================
# Stage 1: Format Training Dataset Builder
# ============================================================

def _discover_all_papers(
    la_dir: Path | None = None,
    txt_dir: Path | None = None,
    max_papers: int | None = None,
) -> List[Tuple[str, str, bool]]:
    """Discover papers from multiple sources.

    Args:
        la_dir: Directory with LaTeX paper subdirectories.
        txt_dir: Directory with plain text .txt papers.
        max_papers: Max total papers across all sources.

    Returns:
        List of (name, content, has_latex) tuples.
    """
    import glob

    papers: List[Tuple[str, str, bool]] = []

    # Source 1: LaTeX papers
    if la_dir and la_dir.is_dir():
        for d in sorted(la_dir.iterdir()):
            if not d.is_dir():
                continue
            tex = d / "latex" / "template.tex"
            if tex.exists():
                papers.append((d.name, tex.read_text(encoding="utf-8"), True))

    # Source 2: Plain text papers (ICLR parsed etc.)
    if txt_dir and txt_dir.is_dir():
        for txt_file in sorted(txt_dir.glob("*.txt")):
            content = txt_file.read_text(encoding="utf-8")
            if len(content.strip()) > 500:  # Skip too-short papers
                papers.append((txt_file.stem, content, False))

    # Shuffle to mix sources
    rng = random.Random(42)
    rng.shuffle(papers)

    if max_papers:
        papers = papers[:max_papers]

    return papers


def _read_paper_text(name: str, content: str, has_latex: bool) -> str:
    """Read paper content; extracts text from LaTeX if needed."""
    if has_latex:
        from ai_scientist.gan_defense.data_utils import extract_text_from_latex_fast
        return extract_text_from_latex_fast(content)
    return content


def build_stage1_dataset(
    papers_dir: Path | None = None,
    text_papers_dir: Path | None = None,
    max_papers: int | None = None,
    verbose: bool = True,
) -> List[dict]:
    """Build Stage 1 dataset: format-diverse clean samples (NO attacks).

    Discovers papers from multiple sources:
      - LaTeX papers (example_papers/)
      - Plain text papers (ICLR parsed .txt)

    For each paper, creates multiple copies with different review JSON targets.
    This teaches the BASE model to output proper JSON format regardless of
    paper content.

    Args:
        papers_dir: Directory containing LaTeX paper subdirectories.
        text_papers_dir: Directory containing plain text .txt papers.
        max_papers: Limit number of papers.
        verbose: Print progress.

    Returns:
        List of dataset entries.
    """
    frontend = SecureFrontend()

    # Resolve default paths
    if papers_dir is None:
        papers_dir = EXAMPLE_PAPERS_DIR
    if text_papers_dir is None:
        _icl_dir = PROJECT / "review_iclr_bench" / "iclr_parsed"
        text_papers_dir = _icl_dir if _icl_dir.is_dir() else None

    papers = _discover_all_papers(papers_dir, text_papers_dir, max_papers)
    n_latex = sum(1 for _, _, hl in papers if hl)
    n_text = len(papers) - n_latex

    if not papers:
        raise ValueError(f"No papers found in {papers_dir} or {text_papers_dir}")

    if verbose:
        print(f"Building Stage 1 dataset from {len(papers)} papers")
        print(f"  LaTeX source: {n_latex} papers")
        print(f"  Text source:  {n_text} papers")
        print(f"  Format variants per paper: {len(STAGE1_REVIEW_VARIANTS)}")
        print(f"  Copies per variant: {STAGE1_CLEAN_COPIES_PER_PAPER}")
        total = len(papers) * len(STAGE1_REVIEW_VARIANTS) * STAGE1_CLEAN_COPIES_PER_PAPER
        print(f"  Expected total: {total:,} entries")
        print()

    dataset = []
    rng = random.Random(42)

    # Score/review templates for diversity
    score_sets = [
        (3, 3, 3, 3, "Reject", "The paper lacks novelty and the experimental validation is insufficient."),
        (5, 5, 5, 5, "Reject", "Marginal contribution with adequate but not compelling results."),
        (7, 7, 7, 7, "Accept", "Solid contribution with well-executed experiments."),
        (8, 8, 8, 8, "Accept", "Strong paper with novel ideas and thorough evaluation."),
        (9, 9, 9, 9, "Accept", "Exceptional work that advances the state of the art significantly."),
        (4, 6, 5, 5, "Reject", "Interesting approach but the theoretical justification is weak."),
        (6, 7, 6, 6, "Accept", "Good technical contribution, though presentation could be improved."),
        (7, 8, 7, 7, "Accept", "Well-motivated problem with a practical and effective solution."),
    ]

    for pi, (name, content, has_latex) in enumerate(papers):
        source_type = "[LaTeX]" if has_latex else "[text]"
        if verbose:
            print(f"[{pi+1}/{len(papers)}] {source_type} {name}")

        try:
            clean_text = _read_paper_text(name, content, has_latex)
        except Exception as e:
            print(f"  Skipping: {e}")
            continue

        # Truncate very long papers to maintain variety without OOM
        max_chars = 8000 if has_latex else 12000
        if len(clean_text) > max_chars:
            # Keep beginning + end to preserve structure
            clean_text = clean_text[:max_chars * 3 // 4] + clean_text[-max_chars // 4:]

        filtered_text = frontend.filter_data(clean_text)

        for variant_idx, variant_template in enumerate(STAGE1_REVIEW_VARIANTS):
            for copy_i in range(STAGE1_CLEAN_COPIES_PER_PAPER):
                base_scores = score_sets[(variant_idx + copy_i) % len(score_sets)]
                novelty, soundness, presentation, overall, decision, review_text = base_scores

                # Add slight random jitter to scores
                def jitter(s):
                    return max(1, min(10, s + rng.randint(-1, 1)))
                novelty = jitter(novelty)
                soundness = jitter(soundness)
                presentation = jitter(presentation)
                overall = jitter(overall)
                decision = "Accept" if overall >= 7 else "Reject"

                review_json = variant_template.format(
                    novelty=novelty, soundness=soundness,
                    presentation=presentation, overall=overall,
                    decision=decision, review=review_text,
                )

                training_text = frontend.encode_clean_review(
                    REVIEW_PROMPT, filtered_text, review_json
                )
                dataset.append({
                    "text": training_text,
                    "type": "clean_format_variant",
                    "paper": name,
                    "payload": f"variant_{variant_idx}_copy_{copy_i}",
                })

    if verbose:
        _print_stats(dataset)

    return dataset


# ============================================================
# Stage 2: Defense Dataset Builder
# ============================================================

def build_stage2_dataset(
    papers_dir: Path | None = None,
    text_papers_dir: Path | None = None,
    max_papers: int | None = None,
    skip_review_generation: bool = False,
    verbose: bool = True,
) -> List[dict]:
    """Build Stage 2 dataset: full V2 defense dataset with all attack types.

    Uses multiple paper sources:
      - LaTeX papers → full attack types (naive/completion/format/boundary)
      - Text papers → completion/format/boundary only (no naive, no LaTeX needed)

    Ratios: 50% clean / 15% naive / 25% completion / 5% format / 5% boundary.

    Args:
        papers_dir: Directory containing LaTeX paper subdirectories.
        text_papers_dir: Directory containing plain text .txt papers.
        max_papers: Limit total papers.
        skip_review_generation: Skip API review (use placeholders).
        verbose: Print progress.

    Returns:
        List of dataset entries.
    """
    from struq_defense.dataset import StruqDatasetBuilder

    builder = StruqDatasetBuilder(seed=42)
    frontend = SecureFrontend()

    # Resolve default paths
    if papers_dir is None:
        papers_dir = EXAMPLE_PAPERS_DIR
    if text_papers_dir is None:
        _icl_dir = PROJECT / "review_iclr_bench" / "iclr_parsed"
        text_papers_dir = _icl_dir if _icl_dir.is_dir() else None

    all_papers = _discover_all_papers(papers_dir, text_papers_dir, max_papers)
    n_latex = sum(1 for _, _, hl in all_papers if hl)
    n_text = len(all_papers) - n_latex

    if not all_papers:
        raise ValueError(f"No papers found in {papers_dir} or {text_papers_dir}")

    n_clean = STAGE2_CLEAN_COPIES_PER_PAPER
    n_naive = STAGE2_NAIVE_VARIANTS_PER_PAPER
    n_compl = STAGE2_COMPLETION_VARIANTS_PER_PAPER
    n_format = STAGE2_FORMAT_VARIANTS_PER_PAPER
    n_boundary = STAGE2_BOUNDARY_VARIANTS_PER_PAPER

    if verbose:
        print(f"Building Stage 2 dataset from {len(all_papers)} papers")
        print(f"  LaTeX source: {n_latex} (full attack types)")
        print(f"  Text source:  {n_text} (completion/format/boundary only)")
        print(f"  Per LaTeX paper: {n_clean}c + {n_naive}na + {n_compl}co "
              f"+ {n_format}f + {n_boundary}b")
        print(f"  Per text paper: {n_clean}c + {n_compl}co + {n_format}f + {n_boundary}b")
        print()

    dataset = []
    rng = random.Random(42)

    for pi, (name, content, has_latex) in enumerate(all_papers):
        source_type = "[LaTeX]" if has_latex else "[text]"
        if verbose:
            print(f"[{pi+1}/{len(all_papers)}] {source_type} {name}")

        # Read paper text
        try:
            clean_text = _read_paper_text(name, content, has_latex)
        except Exception as e:
            print(f"  Skipping: {e}")
            continue

        # Truncate long papers
        max_chars = 8000 if has_latex else 12000
        if len(clean_text) > max_chars:
            clean_text = clean_text[:max_chars * 3 // 4] + clean_text[-max_chars // 4:]

        # For LaTeX papers, keep LaTeX content for naive attack generation
        latex_content = content if has_latex else ""

        # Generate clean review target
        if skip_review_generation:
            clean_review = {"novelty": 5, "soundness": 5, "presentation": 5,
                            "overall": 5, "decision": "Reject",
                            "review": "Standard review."}
            clean_review_json = json.dumps(clean_review, ensure_ascii=False)
        else:
            try:
                if verbose:
                    print("  Generating clean review via API...")
                clean_review = builder.generate_clean_review(clean_text)
                clean_review_json = json.dumps(clean_review, ensure_ascii=False)
                if verbose:
                    print(f"  Clean review: overall={clean_review.get('overall', '?')}")
            except Exception as e:
                print(f"  API review failed: {e}, using placeholder")
                clean_review = {"novelty": 5, "soundness": 5, "presentation": 5,
                                "overall": 5, "decision": "Reject",
                                "review": "API unavailable."}
                clean_review_json = json.dumps(clean_review, ensure_ascii=False)

        # ---- 1) Clean samples ----
        filtered_text = frontend.filter_data(clean_text)
        for _ in range(n_clean):
            training_text = frontend.encode_clean_review(
                REVIEW_PROMPT, filtered_text, clean_review_json
            )
            dataset.append({
                "text": training_text, "type": "clean",
                "paper": name, "payload": "",
            })

        # ---- 2) Naive attack samples (LaTeX only) ----
        if has_latex:
            for vi in range(n_naive):
                try:
                    attacked_text = builder.generate_naive_attack(latex_content, clean_text)
                    filtered_attacked = frontend.filter_data(attacked_text)
                    training_text = frontend.encode_clean_review(
                        REVIEW_PROMPT, filtered_attacked, clean_review_json
                    )
                    dataset.append({
                        "text": training_text, "type": "naive_attack",
                        "paper": name, "payload": f"naive_v{vi}",
                    })
                except Exception as e:
                    if verbose:
                        print(f"  Naive attack {vi} failed: {e}")

        # ---- 3) Completion attack samples ----
        for vi in range(n_compl):
            try:
                if has_latex:
                    attacked_text = builder.generate_completion_attack(latex_content, clean_text)
                else:
                    # Text-only: use built-in completion attack
                    snippet = clean_text[:4000]
                    attacked_text = builder._generate_completion_attack_text(snippet)
                filtered_attacked = frontend.filter_data(attacked_text)
                if rng.random() < STAGE2_NEGATIVE_TARGET_RATIO:
                    target = NEGATIVE_REVIEW_TARGET
                else:
                    target = clean_review_json
                training_text = frontend.encode_clean_review(
                    REVIEW_PROMPT, filtered_attacked, target
                )
                dataset.append({
                    "text": training_text, "type": "completion_attack",
                    "paper": name, "payload": f"completion_v{vi}",
                })
            except Exception as e:
                if verbose:
                    print(f"  Completion attack {vi} failed: {e}")

        # ---- 4) Format attack samples ----
        for vi in range(n_format):
            try:
                attacked_text, attack_desc = _generate_format_attack(clean_text, rng)
                filtered_attacked = frontend.filter_data(attacked_text)
                training_text = frontend.encode_clean_review(
                    REVIEW_PROMPT, filtered_attacked, clean_review_json
                )
                dataset.append({
                    "text": training_text, "type": "format_attack",
                    "paper": name, "payload": attack_desc,
                })
            except Exception as e:
                if verbose:
                    print(f"  Format attack {vi} failed: {e}")

        # ---- 5) Boundary attack samples ----
        for vi in range(n_boundary):
            try:
                attacked_text, attack_desc = _generate_boundary_attack(clean_text, rng)
                filtered_attacked = frontend.filter_data(attacked_text)
                training_text = frontend.encode_clean_review(
                    REVIEW_PROMPT, filtered_attacked, clean_review_json
                )
                dataset.append({
                    "text": training_text, "type": "boundary_attack",
                    "paper": name, "payload": attack_desc,
                })
            except Exception as e:
                if verbose:
                    print(f"  Boundary attack {vi} failed: {e}")

    if verbose:
        _print_stats(dataset)

    return dataset


def _generate_format_attack(clean_text: str, rng: random.Random) -> Tuple[str, str]:
    """Generate a format attack on paper text."""
    n = rng.randint(1, min(3, len(FORMAT_ATTACKS)))
    payloads = rng.sample(FORMAT_ATTACKS, n)
    combined = "\n".join(f"[FORMAT NOTE] {p}" for p in payloads)

    position = rng.choice(["beginning", "middle", "end"])
    text_len = len(clean_text)
    if position == "beginning":
        split = max(100, text_len // 4)
        attacked = clean_text[:split] + "\n" + combined + "\n" + clean_text[split:]
    elif position == "middle":
        split = text_len // 2
        attacked = clean_text[:split] + "\n" + combined + "\n" + clean_text[split:]
    else:
        attacked = clean_text + "\n" + combined

    return attacked, "; ".join(p[:60] for p in payloads)


def _generate_boundary_attack(clean_text: str, rng: random.Random) -> Tuple[str, str]:
    """Generate a boundary attack on paper text."""
    template = rng.choice(BOUNDARY_ATTACK_TEMPLATES)
    if isinstance(template, tuple):
        template = template[0]
    score = rng.randint(7, 10)
    payload = template.replace("{score}", str(score))

    position = rng.choice(["end", "middle", "beginning"])
    text_len = len(clean_text)
    if position == "beginning":
        split = max(100, text_len // 5)
        attacked = clean_text[:split] + "\n" + payload + "\n" + clean_text[split:]
    elif position == "middle":
        split = text_len // 2
        attacked = clean_text[:split] + "\n" + payload + "\n" + clean_text[split:]
    else:
        attacked = clean_text + "\n" + payload

    return attacked, f"boundary_score_{score}"


def _print_stats(dataset: List[dict]):
    total = len(dataset)
    types = {}
    for d in dataset:
        t = d["type"]
        types[t] = types.get(t, 0) + 1
    print()
    print(f"  Dataset: {total} total entries")
    for t, c in sorted(types.items()):
        print(f"    {t:22s}: {c:5d} ({c/total*100:.1f}%)")
    total_chars = sum(len(d["text"]) for d in dataset)
    print(f"    Est. tokens: ~{total_chars // 4:,}")


# ============================================================
# Training Functions
# ============================================================

def _detect_attn_impl() -> str:
    """Detect best available attention implementation for long sequences."""
    try:
        import flash_attn  # noqa: F401
        return "flash_attention_2"
    except ImportError:
        return "sdpa"  # Slower but always available


def _load_base_model_and_tokenizer(model_path: str, verbose: bool = True):
    """Load base model with 4-bit quantization and tokenizer."""
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    if verbose:
        print(f"Loading base model from: {model_path}")

    compute_dtype = getattr(torch, BNB_4BIT_COMPUTE_DTYPE)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=LOAD_IN_4BIT,
        bnb_4bit_quant_type=BNB_4BIT_QUANT_TYPE,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=BNB_4BIT_USE_DOUBLE_QUANT,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation=_detect_attn_impl(),
    )
    model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    _add_special_tokens(tokenizer, model)
    _initialize_special_embeddings(model, tokenizer)

    return model, tokenizer


def _apply_lora(model, r: int, alpha: int, target_modules: list[str], verbose: bool = True):
    """Apply LoRA to the model."""
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    model = prepare_model_for_kbit_training(model)
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    lora_config = LoraConfig(
        r=r, lora_alpha=alpha, lora_dropout=LORA_DROPOUT,
        target_modules=target_modules, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    if verbose:
        model.print_trainable_parameters()
    return model


def _resolve_model_path() -> str:
    """Resolve local model path or fall back to HF Hub."""
    local = LOCAL_MODEL_PATH or os.environ.get("STRUQ_LOCAL_MODEL", "")
    if local and os.path.isdir(local):
        return local
    return BASE_MODEL


# ============================================================
# Stage 1: Format Training
# ============================================================

def train_stage1_format(
    dataset: list[dict] | None = None,
    dataset_path: str | None = None,
    output_dir: str = STAGE1_OUTPUT,
    base_model: str | None = None,
    num_epochs: int = STAGE1_NUM_EPOCHS,
    learning_rate: float = STAGE1_LEARNING_RATE,
    max_seq_length: int = STAGE1_MAX_SEQ_LENGTH,
    lora_r: int = STAGE1_LORA_R,
    lora_alpha: int = STAGE1_LORA_ALPHA,
    resume_from_checkpoint: bool = True,
    verbose: bool = True,
) -> Dict:
    """Stage 1: Format training — teach base model to output JSON reviews.

    Args:
        dataset: In-memory dataset list.
        dataset_path: Path to dataset JSON file.
        output_dir: LoRA adapter save directory.
        base_model: Override base model path.
        num_epochs: Training epochs.
        learning_rate: Learning rate.
        max_seq_length: Max sequence length.
        lora_r: LoRA rank.
        lora_alpha: LoRA alpha.
        resume_from_checkpoint: Resume if checkpoint exists.
        verbose: Print progress.

    Returns:
        Dict with model, tokenizer, output_dir, metrics.
    """
    from transformers import TrainingArguments, Trainer

    print("=" * 60)
    print("  V2_A Stage 1: Format Training")
    print("=" * 60)

    # Load dataset
    if dataset is None:
        if dataset_path is None:
            dataset_path = DATASET_STAGE1_OUTPUT
        with open(dataset_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)

    if verbose:
        print(f"Dataset: {len(dataset)} entries")
        types = {}
        for d in dataset:
            types[d["type"]] = types.get(d["type"], 0) + 1
        for t, c in sorted(types.items()):
            print(f"  {t}: {c}")

    # Load model
    model_path = base_model or _resolve_model_path()
    model, tokenizer = _load_base_model_and_tokenizer(model_path, verbose=verbose)

    # Apply LoRA
    model = _apply_lora(model, lora_r, lora_alpha, STAGE1_LORA_TARGET_MODULES, verbose=verbose)

    # Prepare dataset
    train_dataset = StruqDatasetV2A(dataset, tokenizer, max_length=max_seq_length)

    # Training args
    warmup_steps = max(1, int(STAGE1_WARMUP_RATIO * len(train_dataset) * num_epochs
                              / (STAGE1_PER_DEVICE_BATCH_SIZE * STAGE1_GRADIENT_ACCUMULATION_STEPS)))

    # Compute dtype for bf16 flag
    _compute_dtype = getattr(torch, BNB_4BIT_COMPUTE_DTYPE)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=STAGE1_PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=STAGE1_GRADIENT_ACCUMULATION_STEPS,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=learning_rate,
        max_grad_norm=MAX_GRAD_NORM,
        warmup_steps=warmup_steps,
        weight_decay=WEIGHT_DECAY,
        logging_steps=LOGGING_STEPS,
        save_steps=STAGE1_SAVE_STEPS,
        save_total_limit=2,
        bf16=(_compute_dtype == torch.bfloat16),
        fp16=False,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        report_to="none",
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
    )

    if verbose:
        print(f"\n  Epochs: {num_epochs}")
        print(f"  Batch: {STAGE1_PER_DEVICE_BATCH_SIZE} x {STAGE1_GRADIENT_ACCUMULATION_STEPS}")
        print(f"  LR: {learning_rate}")
        print(f"  Max seq: {max_seq_length}")
        print(f"  LoRA r={lora_r}, alpha={lora_alpha}")
        print(f"  Output: {output_dir}")
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            print(f"  VRAM: {(total-free)/1e9:.1f}GB used / {total/1e9:.1f}GB total")
        print()

    gc.collect()
    torch.cuda.empty_cache()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=_collate_fn,
    )
    trainer.tokenizer = tokenizer

    # Train (resume from checkpoint if exists)
    if resume_from_checkpoint:
        checkpoints = list(Path(output_dir).glob("checkpoint-*"))
        if checkpoints:
            print(f"Resuming from {checkpoints[-1]}")
            trainer.train(resume_from_checkpoint=str(checkpoints[-1]))
        else:
            trainer.train()
    else:
        trainer.train()

    # Save adapter
    print(f"\nSaving Stage 1 adapter to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save struq_config
    struq_config = {
        "special_tokens": SPECIAL_TOKENS,
        "base_model": model_path,
        "structured_query_format": "mark_inst_coln",
        "version": "v2_a",
        "stage": "stage1_format",
        "training_config": {
            "lora_r": lora_r, "lora_alpha": lora_alpha,
            "num_epochs": num_epochs, "learning_rate": learning_rate,
            "max_seq_length": max_seq_length,
        },
    }
    with open(os.path.join(output_dir, "struq_config.json"), "w") as f:
        json.dump(struq_config, f, indent=2)

    metrics = {}
    if trainer.state.log_history:
        metrics["log_history"] = trainer.state.log_history

    print("Stage 1 complete!")
    return {"model": model, "tokenizer": tokenizer,
            "output_dir": output_dir, "metrics": metrics}


# ============================================================
# Stage 2: Defense Training
# ============================================================

def train_stage2_defense(
    dataset: list[dict] | None = None,
    dataset_path: str | None = None,
    stage1_checkpoint: str = STAGE1_OUTPUT,
    output_dir: str = STAGE2_OUTPUT,
    base_model: str | None = None,
    num_epochs: int = STAGE2_NUM_EPOCHS,
    learning_rate: float = STAGE2_LEARNING_RATE,
    max_seq_length: int = STAGE2_MAX_SEQ_LENGTH,
    lora_r: int = STAGE2_LORA_R,
    lora_alpha: int = STAGE2_LORA_ALPHA,
    val_split: float = STAGE2_VAL_SPLIT,
    resume_from_checkpoint: bool = True,
    use_label_masking: bool = True,
    verbose: bool = True,
) -> Dict:
    """Stage 2: Defense training — teach model to ignore [DATA] instructions.

    Continues training from the Stage 1 checkpoint. Loads the Stage 1 LoRA
    adapter and trains on the full V2 defense dataset.

    Args:
        dataset: In-memory dataset list.
        dataset_path: Path to dataset JSON file.
        stage1_checkpoint: Path to Stage 1 LoRA adapter.
        output_dir: Output directory for final adapter.
        base_model: Override base model path.
        num_epochs: Training epochs.
        learning_rate: Learning rate.
        max_seq_length: Max sequence length.
        lora_r: LoRA rank (must match Stage 1).
        lora_alpha: LoRA alpha.
        val_split: Validation split fraction.
        resume_from_checkpoint: Resume if checkpoint exists.
        use_label_masking: Enable label masking (focus loss on [RESP]).
        verbose: Print progress.

    Returns:
        Dict with metrics.
    """
    from transformers import TrainingArguments, Trainer
    from peft import PeftModel

    print("=" * 60)
    print("  V2_A Stage 2: Defense Training")
    print("=" * 60)

    # Load dataset
    if dataset is None:
        if dataset_path is None:
            dataset_path = DATASET_STAGE2_OUTPUT
        with open(dataset_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)

    # Dataset split
    rng = random.Random(42)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)
    val_size = max(1, int(len(dataset) * val_split))
    train_indices = indices[val_size:]
    val_indices = indices[:val_size]
    train_data = [dataset[i] for i in train_indices]
    val_data = [dataset[i] for i in val_indices]

    if verbose:
        print(f"Dataset: {len(train_data)} train / {len(val_data)} validation")
        train_types = {}
        for d in train_data:
            train_types[d["type"]] = train_types.get(d["type"], 0) + 1
        for t, c in sorted(train_types.items()):
            print(f"  train {t}: {c}")
        val_types = {}
        for d in val_data:
            val_types[d["type"]] = val_types.get(d["type"], 0) + 1
        for t, c in sorted(val_types.items()):
            print(f"  val   {t}: {c}")

    # Load base model
    model_path = base_model or _resolve_model_path()
    model, tokenizer = _load_base_model_and_tokenizer(model_path, verbose=verbose)

    # Load Stage 1 LoRA adapter (format-trained weights)
    if verbose:
        print(f"\nLoading Stage 1 adapter from: {stage1_checkpoint}")
    model = PeftModel.from_pretrained(model, stage1_checkpoint, is_trainable=True)

    if verbose:
        model.print_trainable_parameters()

    # Prepare datasets
    train_dataset = StruqDatasetV2A(train_data, tokenizer, max_length=max_seq_length)
    eval_dataset = StruqDatasetV2A(val_data, tokenizer, max_length=max_seq_length)

    # Label masking collate
    collate_fn = create_label_masking_collate(tokenizer) if use_label_masking else None
    if verbose and use_label_masking:
        print("\nLabel masking ENABLED — loss only on [RESP] portion")

    # Training args
    warmup_steps = max(1, int(STAGE2_WARMUP_RATIO * len(train_dataset) * num_epochs
                              / (STAGE2_PER_DEVICE_BATCH_SIZE * STAGE2_GRADIENT_ACCUMULATION_STEPS)))

    _compute_dtype = getattr(torch, BNB_4BIT_COMPUTE_DTYPE)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=STAGE2_PER_DEVICE_BATCH_SIZE,
        per_device_eval_batch_size=STAGE2_PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=STAGE2_GRADIENT_ACCUMULATION_STEPS,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=learning_rate,
        max_grad_norm=MAX_GRAD_NORM,
        warmup_steps=warmup_steps,
        weight_decay=WEIGHT_DECAY,
        logging_steps=LOGGING_STEPS,
        eval_strategy="steps",
        eval_steps=STAGE2_EVAL_STEPS,
        save_steps=STAGE2_SAVE_STEPS,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=(_compute_dtype == torch.bfloat16),
        fp16=False,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        report_to="none",
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
    )

    if verbose:
        print(f"\n  Base model: {model_path}")
        print(f"  Stage 1 checkpoint: {stage1_checkpoint}")
        print(f"  Epochs: {num_epochs}")
        print(f"  Batch: {STAGE2_PER_DEVICE_BATCH_SIZE} x {STAGE2_GRADIENT_ACCUMULATION_STEPS}")
        print(f"  LR: {learning_rate}")
        print(f"  Max seq: {max_seq_length}")
        print(f"  LoRA r={lora_r}, alpha={lora_alpha}")
        print(f"  Output: {output_dir}")
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            print(f"  VRAM: {(total-free)/1e9:.1f}GB used / {total/1e9:.1f}GB total")
        print()

    gc.collect()
    torch.cuda.empty_cache()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_fn,
    )
    trainer.tokenizer = tokenizer

    # Train
    if resume_from_checkpoint:
        checkpoints = list(Path(output_dir).glob("checkpoint-*"))
        if checkpoints:
            print(f"Resuming from {checkpoints[-1]}")
            trainer.train(resume_from_checkpoint=str(checkpoints[-1]))
        else:
            trainer.train()
    else:
        trainer.train()

    # Save final adapter
    print(f"\nSaving Stage 2 adapter to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save struq_config
    struq_config = {
        "special_tokens": SPECIAL_TOKENS,
        "base_model": model_path,
        "structured_query_format": "mark_inst_coln",
        "version": "v2_a",
        "stage": "stage2_defense",
        "label_masking": use_label_masking,
        "training_config": {
            "lora_r": lora_r, "lora_alpha": lora_alpha,
            "num_epochs": num_epochs, "learning_rate": learning_rate,
            "max_seq_length": max_seq_length,
        },
    }
    with open(os.path.join(output_dir, "struq_config.json"), "w") as f:
        json.dump(struq_config, f, indent=2)

    # Metrics
    metrics = {}
    if trainer.state.log_history:
        metrics["log_history"] = trainer.state.log_history
        eval_losses = [e for e in trainer.state.log_history if "eval_loss" in e]
        if eval_losses:
            best = min(eval_losses, key=lambda x: x["eval_loss"])
            metrics["best_eval_loss"] = best["eval_loss"]
            metrics["best_eval_step"] = best.get("step", "?")

    print("Stage 2 complete!")
    if "best_eval_loss" in metrics:
        print(f"  Best eval loss: {metrics['best_eval_loss']:.4f} "
              f"(step {metrics['best_eval_step']})")

    return {"model": model, "tokenizer": tokenizer,
            "output_dir": output_dir, "metrics": metrics}


# ============================================================
# Convenience: Run both stages
# ============================================================

def train_v2_a_full_pipeline(
    papers_dir: Path | None = None,
    max_papers: int | None = None,
    skip_review_generation: bool = False,
    verbose: bool = True,
) -> Dict:
    """Run both Stage 1 and Stage 2 training in sequence.

    Args:
        papers_dir: Directory containing paper subdirectories.
        max_papers: Limit number of papers.
        skip_review_generation: Skip API review generation.
        verbose: Print progress.

    Returns:
        Dict with Stage 1 and Stage 2 results.
    """
    # Stage 1
    print("\n" + "=" * 60)
    print("  V2_A Full Pipeline")
    print("=" * 60)

    # Build Stage 1 dataset
    print("\n[1/4] Building Stage 1 dataset...")
    stage1_dataset = build_stage1_dataset(papers_dir, max_papers, verbose)
    os.makedirs(os.path.dirname(DATASET_STAGE1_OUTPUT), exist_ok=True)
    with open(DATASET_STAGE1_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(stage1_dataset, f, ensure_ascii=False, indent=2)
    print(f"Saved to {DATASET_STAGE1_OUTPUT}")

    # Train Stage 1
    print("\n[2/4] Training Stage 1 (format)...")
    stage1_result = train_stage1_format(dataset=stage1_dataset, verbose=verbose)

    # Build Stage 2 dataset
    print("\n[3/4] Building Stage 2 dataset...")
    stage2_dataset = build_stage2_dataset(papers_dir, max_papers, skip_review_generation, verbose)
    os.makedirs(os.path.dirname(DATASET_STAGE2_OUTPUT), exist_ok=True)
    with open(DATASET_STAGE2_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(stage2_dataset, f, ensure_ascii=False, indent=2)
    print(f"Saved to {DATASET_STAGE2_OUTPUT}")

    # Train Stage 2
    print("\n[4/4] Training Stage 2 (defense)...")
    stage2_result = train_stage2_defense(dataset=stage2_dataset, verbose=verbose)

    print("\n" + "=" * 60)
    print("  V2_A Pipeline Complete!")
    print(f"  Stage 1 adapter: {stage1_result['output_dir']}")
    print(f"  Stage 2 adapter: {stage2_result['output_dir']}")
    print("=" * 60)

    return {"stage1": stage1_result, "stage2": stage2_result}


# ============================================================
# CLI
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="V2_A Two-Stage Training")
    subparsers = parser.add_subparsers(dest="cmd", help="Subcommands")

    # stage1
    p1 = subparsers.add_parser("stage1", help="Run Stage 1 format training")
    p1.add_argument("--dataset", type=str, default=None, help="Dataset JSON path")
    p1.add_argument("--papers", type=str, default=None, help="Papers directory")
    p1.add_argument("--max-papers", type=int, default=None)
    p1.add_argument("--output", type=str, default=STAGE1_OUTPUT)
    p1.add_argument("--model", type=str, default=None, help="Base model path override")
    p1.add_argument("--epochs", type=int, default=STAGE1_NUM_EPOCHS)
    p1.add_argument("--lr", type=float, default=STAGE1_LEARNING_RATE)
    p1.add_argument("--no-resume", action="store_true")
    p1.add_argument("--no-label-masking", action="store_true")

    # stage2
    p2 = subparsers.add_parser("stage2", help="Run Stage 2 defense training")
    p2.add_argument("--dataset", type=str, default=None, help="Dataset JSON path")
    p2.add_argument("--papers", type=str, default=None, help="Papers directory")
    p2.add_argument("--max-papers", type=int, default=None)
    p2.add_argument("--stage1-checkpoint", type=str, default=STAGE1_OUTPUT)
    p2.add_argument("--output", type=str, default=STAGE2_OUTPUT)
    p2.add_argument("--model", type=str, default=None, help="Base model path override")
    p2.add_argument("--epochs", type=int, default=STAGE2_NUM_EPOCHS)
    p2.add_argument("--lr", type=float, default=STAGE2_LEARNING_RATE)
    p2.add_argument("--no-resume", action="store_true")
    p2.add_argument("--no-label-masking", action="store_true")
    p2.add_argument("--skip-reviews", action="store_true")

    # pipeline
    p3 = subparsers.add_parser("pipeline", help="Run full two-stage pipeline")
    p3.add_argument("--papers", type=str, default=None)
    p3.add_argument("--max-papers", type=int, default=None)
    p3.add_argument("--skip-reviews", action="store_true")

    args = parser.parse_args()

    if args.cmd == "stage1":
        dataset = None
        if args.dataset:
            with open(args.dataset, "r", encoding="utf-8") as f:
                dataset = json.load(f)
        elif args.papers:
            print("Building Stage 1 dataset...")
            dataset = build_stage1_dataset(
                Path(args.papers) if not Path(args.papers).is_absolute()
                else Path(args.papers),
                args.max_papers,
            )
            os.makedirs(os.path.dirname(DATASET_STAGE1_OUTPUT), exist_ok=True)
            with open(DATASET_STAGE1_OUTPUT, "w", encoding="utf-8") as f:
                json.dump(dataset, f, ensure_ascii=False, indent=2)
        train_stage1_format(
            dataset=dataset, dataset_path=args.dataset,
            output_dir=args.output, base_model=args.model,
            num_epochs=args.epochs, learning_rate=args.lr,
            resume_from_checkpoint=not args.no_resume,
        )

    elif args.cmd == "stage2":
        dataset = None
        if args.dataset:
            with open(args.dataset, "r", encoding="utf-8") as f:
                dataset = json.load(f)
        elif args.papers:
            print("Building Stage 2 dataset...")
            dataset = build_stage2_dataset(
                Path(args.papers) if not Path(args.papers).is_absolute()
                else Path(args.papers),
                args.max_papers, args.skip_reviews,
            )
            os.makedirs(os.path.dirname(DATASET_STAGE2_OUTPUT), exist_ok=True)
            with open(DATASET_STAGE2_OUTPUT, "w", encoding="utf-8") as f:
                json.dump(dataset, f, ensure_ascii=False, indent=2)
        train_stage2_defense(
            dataset=dataset, dataset_path=args.dataset,
            stage1_checkpoint=args.stage1_checkpoint,
            output_dir=args.output, base_model=args.model,
            num_epochs=args.epochs, learning_rate=args.lr,
            resume_from_checkpoint=not args.no_resume,
            use_label_masking=not args.no_label_masking,
        )

    elif args.cmd == "pipeline":
        train_v2_a_full_pipeline(
            papers_dir=Path(args.papers) if args.papers else None,
            max_papers=args.max_papers,
            skip_review_generation=args.skip_reviews,
        )

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
