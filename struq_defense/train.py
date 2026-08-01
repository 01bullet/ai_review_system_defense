"""
Structured Instruction Tuning — fine-tune a base LLM to support structured queries.

Trains a base (non-instruction-tuned) LLM to:
1. Follow review instructions ONLY in the [INST] portion
2. Ignore any instructions hidden in the [DATA] portion
3. Output reviews in the [RESP] section

Uses QLoRA (4-bit quantized LoRA) for memory-efficient fine-tuning.
Feasible on a single GPU with ≥16GB VRAM.

Reference:
  - StruQ Section 4.4 "Structured Instruction Tuning"
  - StruQ Algorithm 1 — dataset construction
"""

from __future__ import annotations

import gc
import json
import os
import sys
from pathlib import Path
from typing import Optional, Dict

# HF mirror for China mainland — must be set BEFORE importing transformers
if "HF_ENDPOINT" not in os.environ:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
from torch.utils.data import Dataset, DataLoader

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

# AutoDL mode — uses expanded config
_AUTODL = os.environ.get("STRUQ_AUTODL", "").lower() in ("1", "true")
if _AUTODL:
    from struq_defense.config_autodl import (
        LORA_R, LORA_ALPHA, LORA_DROPOUT, LORA_TARGET_MODULES,
        LOAD_IN_4BIT, BNB_4BIT_COMPUTE_DTYPE, BNB_4BIT_QUANT_TYPE, BNB_4BIT_USE_DOUBLE_QUANT,
        NUM_EPOCHS, LEARNING_RATE, PER_DEVICE_BATCH_SIZE, GRADIENT_ACCUMULATION_STEPS,
        MAX_SEQ_LENGTH, WARMUP_RATIO, WEIGHT_DECAY, MAX_GRAD_NORM,
        SAVE_STEPS, LOGGING_STEPS,
        LORA_OUTPUT, MERGED_OUTPUT, DATASET_OUTPUT,
    )
else:
    from struq_defense.config import (
        LORA_R, LORA_ALPHA, LORA_DROPOUT, LORA_TARGET_MODULES,
        LOAD_IN_4BIT, BNB_4BIT_COMPUTE_DTYPE, BNB_4BIT_QUANT_TYPE, BNB_4BIT_USE_DOUBLE_QUANT,
        NUM_EPOCHS, LEARNING_RATE, PER_DEVICE_BATCH_SIZE, GRADIENT_ACCUMULATION_STEPS,
        MAX_SEQ_LENGTH, WARMUP_RATIO, WEIGHT_DECAY, MAX_GRAD_NORM,
        SAVE_STEPS, LOGGING_STEPS,
        LORA_OUTPUT, MERGED_OUTPUT, DATASET_OUTPUT,
    )
from struq_defense.config import (
    BASE_MODEL,
    LOCAL_MODEL_PATH,
    SPECIAL_TOKENS,
    DEVICE,
)
from struq_defense.frontend import SecureFrontend


# ---- Dataset wrapper ----

class StruqDataset(Dataset):
    """PyTorch Dataset wrapping the StruQ training data.

    Each item is a full training text (structured query + desired response).
    The tokenizer pads/truncates to MAX_SEQ_LENGTH.
    """

    def __init__(self, data: list[dict], tokenizer, max_length: int = MAX_SEQ_LENGTH):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        text = self.data[idx]["text"]

        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        # For causal LM training, labels = input_ids
        input_ids = encoded["input_ids"][0]
        attention_mask = encoded["attention_mask"][0]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone(),
        }


# ---- Training utilities ----

def _add_special_tokens(tokenizer, model):
    """Add StruQ special tokens to tokenizer and resize model embeddings.

    Args:
        tokenizer: HuggingFace tokenizer.
        model: HuggingFace model.

    Returns:
        Number of tokens added.
    """
    num_added = tokenizer.add_tokens(SPECIAL_TOKENS)
    if num_added > 0:
        model.resize_token_embeddings(len(tokenizer))
        print(f"  Added {num_added} special tokens, "
              f"vocab size: {len(tokenizer)}")
    return num_added


def _initialize_special_embeddings(model, tokenizer):
    """Initialize special token embeddings from similar tokens.

    Following StruQ Section 4.3 — this is CRITICAL for utility.
    Without proper initialization, the model cannot learn useful
    representations for new tokens during fine-tuning.
    """
    frontend = SecureFrontend()
    frontend.initialize_special_embeddings(model, tokenizer)

    # Verify
    for special_tok, source_text in frontend.embed_init_map.items():
        tok_id = tokenizer.convert_tokens_to_ids(special_tok)
        if tok_id != tokenizer.unk_token_id:
            source_id = tokenizer.encode(source_text, add_special_tokens=False)[0]
            cos = torch.cosine_similarity(
                model.get_input_embeddings().weight[tok_id].unsqueeze(0),
                model.get_input_embeddings().weight[source_id].unsqueeze(0),
            )
            print(f"  Embed init: {special_tok} ← {source_text} (cos_sim={cos.item():.4f})")


def _collate_fn(batch):
    """Collate batch of tokenized sequences."""
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
    }


# ---- Main training function ----

def train_structured_instruction_tuning(
    dataset_path: str | None = None,
    dataset: list[dict] | None = None,
    base_model: str = BASE_MODEL,
    output_dir: str = LORA_OUTPUT,
    num_epochs: int = NUM_EPOCHS,
    learning_rate: float = LEARNING_RATE,
    resume_from_checkpoint: bool = True,
    verbose: bool = True,
) -> Dict:
    """Run structured instruction tuning (QLoRA).

    Args:
        dataset_path: Path to JSON dataset file.
        dataset: In-memory dataset (alternative to dataset_path).
        base_model: HuggingFace model identifier (must be a base model).
        output_dir: Directory to save LoRA adapter.
        num_epochs: Number of training epochs.
        learning_rate: Learning rate for LoRA parameters.
        resume_from_checkpoint: Resume if checkpoint exists.
        verbose: Print progress.

    Returns:
        Dict with training metrics.
    """
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        BitsAndBytesConfig,
        TrainingArguments,
        Trainer,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    import transformers

    # ---- Load dataset ----
    if dataset is None:
        if dataset_path is None:
            dataset_path = DATASET_OUTPUT
        with open(dataset_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)

    if verbose:
        print(f"Loaded dataset: {len(dataset)} entries")
        types = {}
        for d in dataset:
            types[d["type"]] = types.get(d["type"], 0) + 1
        for t, c in sorted(types.items()):
            print(f"  {t}: {c}")

    # ---- Resolve model path (local > HF Hub) ----
    local_path = LOCAL_MODEL_PATH or os.environ.get("STRUQ_LOCAL_MODEL", "")
    if local_path and os.path.isdir(local_path):
        model_path = local_path
        if verbose:
            print(f"\nLoading base model from local: {model_path}")
    else:
        model_path = base_model
        if verbose:
            print(f"\nLoading base model from HF Hub: {model_path}")
        if local_path:
            print(f"  Warning: LOCAL_MODEL_PATH={local_path} not found, falling back to HF Hub")

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
    )
    model.config.use_cache = False  # Required for gradient checkpointing

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )

    # Ensure pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    # ---- Add special tokens ----
    _add_special_tokens(tokenizer, model)
    _initialize_special_embeddings(model, tokenizer)

    # ---- Prepare for k-bit training ----
    model = prepare_model_for_kbit_training(model)
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    # ---- LoRA setup ----
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    if verbose:
        model.print_trainable_parameters()

    # ---- Prepare training data ----
    train_dataset = StruqDataset(dataset, tokenizer)
    # Use custom collate_fn to avoid transformers version-dependent API
    # DataCollatorForLanguageModeling has broken tokenizer/processing_class param names
    # across versions. Our dataset already returns labels, so a simple collate is safer.

    # ---- Training arguments ----
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=learning_rate,
        max_grad_norm=MAX_GRAD_NORM,
        warmup_steps=max(1, int(WARMUP_RATIO * len(train_dataset) * num_epochs
                                / (PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))),
        weight_decay=WEIGHT_DECAY,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=2,
        bf16=(compute_dtype == torch.bfloat16),
        fp16=False,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        report_to="none",
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
    )

    # ---- Train ----
    if verbose:
        print(f"\nStarting structured instruction tuning...")
        print(f"  Epochs: {num_epochs}")
        print(f"  Batch size: {PER_DEVICE_BATCH_SIZE} × {GRADIENT_ACCUMULATION_STEPS}")
        print(f"  LR: {learning_rate}")
        print(f"  Max seq length: {MAX_SEQ_LENGTH}")
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            used = total - free
            print(f"  VRAM: {used/1e9:.1f}GB used / {total/1e9:.1f}GB total ({free/1e9:.1f}GB free)")
        print()

    # Aggressive VRAM cleanup before training
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        if verbose:
            print(f"  VRAM after cleanup: {(total-free)/1e9:.1f}GB used / {total/1e9:.1f}GB total ({free/1e9:.1f}GB free)")
        if free < 0.5e9:
            print(f"  WARNING: Less than 0.5GB free VRAM — training may OOM")
        print()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=_collate_fn,
    )
    trainer.tokenizer = tokenizer

    # Train (resume from checkpoint if available)
    if resume_from_checkpoint:
        checkpoints = list(Path(output_dir).glob("checkpoint-*"))
        if checkpoints:
            if verbose:
                print(f"Resuming from {checkpoints[-1]}")
            trainer.train(resume_from_checkpoint=str(checkpoints[-1]))
        else:
            trainer.train()
    else:
        trainer.train()

    # ---- Save ----
    if verbose:
        print(f"\nSaving LoRA adapter to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save the special tokens configuration
    struq_config = {
        "special_tokens": SPECIAL_TOKENS,
        "base_model": base_model,
        "structured_query_format": "mark_inst_coln",
    }
    with open(os.path.join(output_dir, "struq_config.json"), "w") as f:
        json.dump(struq_config, f, indent=2)

    # ---- Extract training metrics ----
    metrics = {}
    if trainer.state.log_history:
        metrics["log_history"] = trainer.state.log_history

    if verbose:
        print("Training complete!")

    return {
        "model": model,
        "tokenizer": tokenizer,
        "output_dir": output_dir,
        "metrics": metrics,
    }


def merge_and_save(
    model,
    tokenizer,
    output_dir: str = MERGED_OUTPUT,
):
    """Merge LoRA adapter with base model and save full weights.

    This produces a standalone model that can be loaded without PEFT.

    Args:
        model: The PEFT model (from train_structured_instruction_tuning).
        tokenizer: The tokenizer (with special tokens).
        output_dir: Directory to save merged model.

    Returns:
        Path to saved model directory.
    """
    import torch
    from peft import PeftModel
    import shutil

    # If the model is a PEFT model, merge and unload
    if hasattr(model, "merge_and_unload"):
        print("  Merging LoRA weights...")
        merged = model.merge_and_unload()
    else:
        merged = model

    # Strip quantization artifacts from the merged model.
    # When a 4-bit loaded model is merged, the bnb Linear4bit modules
    # retain their quantization metadata as persistent buffers.
    # save_pretrained() calls state_dict() internally and re-adds them,
    # so we manually save via safetensors to keep only clean weight keys.
    if hasattr(merged, 'config') and hasattr(merged.config, 'quantization_config'):
        del merged.config.quantization_config

    state_dict = merged.state_dict()
    quant_suffixes = ('quant_state', 'nested_absmax', 'nested_quant_map',
                      'quant_map', 'absmax', 'bitsandbytes')
    clean_dict = {k: v for k, v in state_dict.items()
                  if not any(s in k for s in quant_suffixes)}
    removed = len(state_dict) - len(clean_dict)
    if removed:
        print(f"  Removed {removed} quantization artifact keys from state dict")

    os.makedirs(output_dir, exist_ok=True)

    if verbose := True:
        print(f"Saving merged model to {output_dir}...")

    # Save manually via safetensors to avoid save_pretrained re-adding quant keys
    from safetensors.torch import save_file
    import json

    # Split large state_dict into shards
    max_shard_bytes = 4 * 1024 ** 3  # 4GB
    shards = []
    current_shard = {}
    current_bytes = 0
    weight_map = {}

    for key, tensor in clean_dict.items():
        # Estimate tensor size in safetensors (same as tensor.nbytes + small overhead)
        tensor_bytes = tensor.nbytes + len(key)
        if current_bytes + tensor_bytes > max_shard_bytes and current_shard:
            shards.append(current_shard)
            current_shard = {}
            current_bytes = 0
        current_shard[key] = tensor
        current_bytes += tensor_bytes
    if current_shard:
        shards.append(current_shard)

    total_shards = len(shards)
    for i, shard in enumerate(shards, 1):
        shard_name = f"model-{i:05d}-of-{total_shards:05d}.safetensors"
        for key in shard:
            weight_map[key] = shard_name
        save_file(shard, os.path.join(output_dir, shard_name))
        if verbose:
            print(f"  Saved {shard_name} ({len(shard)} tensors, "
                  f"{sum(t.nbytes for t in shard.values())/1e9:.1f} GB)")

    # Save index
    index = {"metadata": {"total_size": sum(t.nbytes for t in clean_dict.values())},
             "weight_map": weight_map}
    with open(os.path.join(output_dir, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f, indent=2)

    # Save config
    merged.config.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Copy struq config
    for src in Path(model.config.output_dir if hasattr(model.config, 'output_dir')
                    else LORA_OUTPUT).glob("struq_config.json"):
        shutil.copy2(src, os.path.join(output_dir, "struq_config.json"))

    return output_dir


# ---- V2 Training Function ----

def train_structured_instruction_tuning_v2(
    dataset: list[dict],
    base_model: str = "Qwen/Qwen2.5-7B-Instruct",
    output_dir: str | None = None,
    num_epochs: int = 8,
    learning_rate: float = 2e-4,
    per_device_batch_size: int = 2,
    gradient_accumulation_steps: int = 4,
    max_seq_length: int = 2048,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_target_modules: list[str] | None = None,
    val_split: float = 0.10,
    resume_from_checkpoint: bool = True,
    use_label_masking: bool = True,
    use_augmentation: bool = False,
    verbose: bool = True,
) -> Dict:
    """V2 structured instruction tuning with label masking + validation.

    Key improvements over V1:
      - Label masking: only compute loss on [RESP] portion (not prompt/data)
      - Validation split: monitor overfitting, save best checkpoint
      - Instruct model support: leverages built-in instruction following
      - Data augmentation: per-epoch attack sample variation (optional)

    Args:
        dataset: List of dataset entries (from build_v2).
        base_model: HuggingFace model identifier.
        output_dir: LoRA adapter save directory.
        num_epochs: Number of training epochs.
        learning_rate: Learning rate.
        per_device_batch_size: Per-GPU batch size.
        gradient_accumulation_steps: Gradient accumulation steps.
        max_seq_length: Max sequence length.
        lora_r: LoRA rank.
        lora_alpha: LoRA alpha.
        lora_target_modules: LoRA target modules.
        val_split: Fraction of data for validation.
        resume_from_checkpoint: Resume if checkpoint exists.
        use_label_masking: Enable label masking (V2 core feature).
        use_augmentation: Enable per-epoch data augmentation.
        verbose: Print progress.

    Returns:
        Dict with training metrics.
    """
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        BitsAndBytesConfig,
        TrainingArguments,
        Trainer,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from struq_defense.dataset import create_label_masking_collate

    if lora_target_modules is None:
        lora_target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]

    if output_dir is None:
        output_dir = LORA_OUTPUT

    # ---- Resolve model path ----
    local_path = LOCAL_MODEL_PATH or os.environ.get("STRUQ_LOCAL_MODEL_V2", "")
    if not local_path:
        local_path = os.environ.get("STRUQ_LOCAL_MODEL", "")
    if local_path and os.path.isdir(local_path):
        model_path = local_path
        if verbose:
            print(f"\nLoading base model from local: {model_path}")
    else:
        model_path = base_model
        if verbose:
            print(f"\nLoading base model from HF Hub: {model_path}")

    # ---- Dataset split ----
    import random
    rng = random.Random(42)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)
    val_size = max(1, int(len(dataset) * val_split))
    train_indices = indices[val_size:]
    val_indices = indices[:val_size]
    train_data = [dataset[i] for i in train_indices]
    val_data = [dataset[i] for i in val_indices]

    if verbose:
        print(f"\nDataset: {len(train_data)} train / {len(val_data)} validation")
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

    # ---- Load model ----
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
    )
    model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    # ---- Add special tokens + initialize embeddings ----
    _add_special_tokens(tokenizer, model)
    _initialize_special_embeddings(model, tokenizer)

    # ---- Prepare for k-bit training ----
    model = prepare_model_for_kbit_training(model)
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    # ---- LoRA setup ----
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=LORA_DROPOUT,
        target_modules=lora_target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    if verbose:
        model.print_trainable_parameters()

    # ---- Prepare datasets ----
    train_dataset = StruqDataset(train_data, tokenizer, max_length=max_seq_length)
    eval_dataset = StruqDataset(val_data, tokenizer, max_length=max_seq_length)

    # V2: Use label masking collate
    collate_fn = create_label_masking_collate(tokenizer) if use_label_masking else None
    if verbose and use_label_masking:
        print("\n[V2] Label masking ENABLED — loss only on [RESP] portion")

    # ---- Training arguments ----
    warmup_steps = max(1, int(WARMUP_RATIO * len(train_dataset) * num_epochs
                              / (per_device_batch_size * gradient_accumulation_steps)))

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=learning_rate,
        max_grad_norm=MAX_GRAD_NORM,
        warmup_steps=warmup_steps,
        weight_decay=WEIGHT_DECAY,
        logging_steps=LOGGING_STEPS,
        eval_strategy="steps",
        eval_steps=200,
        save_steps=200,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=(compute_dtype == torch.bfloat16),
        fp16=False,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        report_to="none",
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
    )

    if verbose:
        print(f"\nStarting V2 structured instruction tuning...")
        print(f"  Base model: {base_model}")
        print(f"  Epochs: {num_epochs}")
        print(f"  Batch size: {per_device_batch_size} × {gradient_accumulation_steps}")
        print(f"  Effective batch: {per_device_batch_size * gradient_accumulation_steps}")
        print(f"  LR: {learning_rate}")
        print(f"  Max seq length: {max_seq_length}")
        print(f"  LoRA r={lora_r}, alpha={lora_alpha}")
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            used = total - free
            print(f"  VRAM: {used/1e9:.1f}GB used / {total/1e9:.1f}GB total "
                  f"({free/1e9:.1f}GB free)")
        print()

    # Aggressive VRAM cleanup
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
            if verbose:
                print(f"Resuming from {checkpoints[-1]}")
            trainer.train(resume_from_checkpoint=str(checkpoints[-1]))
        else:
            trainer.train()
    else:
        trainer.train()

    # ---- Save ----
    if verbose:
        print(f"\nSaving LoRA adapter to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save V2 config
    struq_config = {
        "special_tokens": SPECIAL_TOKENS,
        "base_model": base_model,
        "structured_query_format": "mark_inst_coln",
        "version": "v2",
        "label_masking": use_label_masking,
        "training_config": {
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
            "num_epochs": num_epochs,
            "learning_rate": learning_rate,
            "max_seq_length": max_seq_length,
        },
    }
    with open(os.path.join(output_dir, "struq_config.json"), "w") as f:
        json.dump(struq_config, f, indent=2)

    # ---- Extract metrics ----
    metrics = {}
    if trainer.state.log_history:
        metrics["log_history"] = trainer.state.log_history
        # Find best eval loss
        eval_losses = [e for e in trainer.state.log_history if "eval_loss" in e]
        if eval_losses:
            best = min(eval_losses, key=lambda x: x["eval_loss"])
            metrics["best_eval_loss"] = best["eval_loss"]
            metrics["best_eval_step"] = best.get("step", "?")

    if verbose:
        print("V2 training complete!")
        if "best_eval_loss" in metrics:
            print(f"  Best eval loss: {metrics['best_eval_loss']:.4f} "
                  f"(step {metrics['best_eval_step']})")

    return {
        "model": model,
        "tokenizer": tokenizer,
        "output_dir": output_dir,
        "metrics": metrics,
    }
