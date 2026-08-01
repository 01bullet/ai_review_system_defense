"""
Manual model loader for Qwen2.5-7B + 4-bit quantization on low-RAM Windows.

Problem: from_pretrained() segfaults on Python 3.13 + transformers 5.x +
Windows + RTX 5060 when using device_map='auto' (real device allocation).

Solution: Load model on meta device (works!), then individually load each
weight tensor from safetensors and assign it directly to the model parameter.

Usage:
    from manual_model_loader import manual_load_4bit
    model, tokenizer = manual_load_4bit("models/qwen2.5-7b")
"""

import os
import gc
import torch
import torch.nn as nn
from safetensors import safe_open
from accelerate import init_empty_weights
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import bitsandbytes as bnb


def manual_load_4bit(
    model_path: str,
    verbose: bool = True,
):
    """
    Load Qwen2.5-7B in 4-bit quantization on low-RAM Windows.

    Strategy:
    1. Load model on meta device with BitsAndBytesConfig (4-bit layers)
    2. Iterate safetensors weights one at a time
    3. For each weight, create real storage on GPU and assign the data
    4. Garbage collect after each batch to keep RAM low
    """
    # ---- Auto-download base model if missing ----
    if not os.path.isdir(model_path) or not any(
        f.endswith(".safetensors") for f in os.listdir(model_path) if os.path.isfile(os.path.join(model_path, f))
    ):
        from ensure_model import ensure_base_model
        ensure_base_model(model_path, verbose=verbose)

    if verbose:
        print(f"Manual 4-bit loading from: {model_path}")

    # Check GPU
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        gpu_name = torch.cuda.get_device_name(0)
        if verbose:
            print(f"  GPU: {gpu_name} ({total_vram:.1f} GB VRAM)")
    target_device = "cuda:0" if use_cuda else "cpu"

    # ---- Step 1: Load model on meta device with 4-bit layers ----
    if verbose:
        print("  Loading model on meta device (4-bit layers)...")

    # IMPORTANT: Do NOT use from_pretrained() here — even with
    # device_map="meta" (without quantization_config) it can segfault on
    # RTX 5060 + Python 3.13.  Instead, build the model directly from config
    # using PyTorch's native meta device (NOT accelerate's init_empty_weights,
    # which adds hooks that can also trigger the crash on this hardware).
    if verbose:
        print("  Loading config...")
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    if verbose:
        print("  Building model on meta device...")
    with torch.device('meta'):
        model = AutoModelForCausalLM.from_config(
            config,
            trust_remote_code=False,
            torch_dtype=torch.bfloat16,
        )

    # Replace nn.Linear with bnb.nn.Linear4bit for 4-bit inference.
    # This must happen BEFORE building param_lookup so it sees Params4bit params.
    n_replaced = _replace_linear_with_bnb_4bit(model, verbose=verbose)
    model.is_loaded_in_4bit = True  # So PeftModel / generation utils detect correctly
    if verbose:
        print(f"  Replaced {n_replaced} Linear layers with Linear4bit")

    total_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"  Loaded: {total_params/1e9:.2f}B params on meta device")

    # ---- Step 2: Collect safetensors files ----
    sf_files = sorted([
        os.path.join(model_path, f)
        for f in os.listdir(model_path)
        if f.endswith(".safetensors")
    ])
    if not sf_files:
        # Try index file
        import json
        idx = os.path.join(model_path, "model.safetensors.index.json")
        if os.path.exists(idx):
            with open(idx) as f:
                index = json.load(f)
            sf_files = sorted(set(
                os.path.join(model_path, wm)
                for wm in index.get("weight_map", {}).values()
            ))

    if verbose:
        print(f"  Found {len(sf_files)} safetensors file(s)")

    # Build parameter name → (module_path, param_name) mapping
    # We need this to assign weights through the module hierarchy
    # (params4bit require special handling)
    param_lookup = {}
    for name, param in model.named_parameters():
        # Convert "model.layers.0.self_attn.q_proj.weight" → the module path
        if "." in name:
            *mod_parts, attr = name.split(".")
            param_lookup[name] = (".".join(mod_parts), attr)
        else:
            param_lookup[name] = ("", name)

    # Build safetensors key → model param mapping
    sf_key_mapping = {}
    for sf_file in sf_files:
        with safe_open(sf_file, framework="pt", device="cpu") as sf:
            for key in sf.keys():
                if "quant_state" in key:
                    continue
                # Map safetensors key to model parameter name
                # Try exact match first, then with/without "model." prefix
                if key in param_lookup:
                    sf_key_mapping[key] = key
                elif key.startswith("model.") and key[6:] in param_lookup:
                    sf_key_mapping[key] = key[6:]
                elif not key.startswith("model.") and f"model.{key}" in param_lookup:
                    sf_key_mapping[key] = f"model.{key}"

    if verbose:
        print(f"  Mapped {len(sf_key_mapping)} safetensors keys to model params")

    # ---- Step 3: Load and assign weights one at a time ----
    if verbose:
        print("  Assigning weights to GPU (one tensor at a time)...")

    assigned = 0
    skipped = 0
    batch_size = 20  # How many tensors to process between GC calls

    for sf_file in sf_files:
        fname = os.path.basename(sf_file)
        if verbose:
            print(f"    Processing {fname}...")

        with safe_open(sf_file, framework="pt", device="cpu") as sf:
            sf_keys = [k for k in sf.keys() if "quant_state" not in k]

            for i, key in enumerate(sf_keys):
                # Get the model parameter name
                param_name = sf_key_mapping.get(key)
                if param_name is None:
                    skipped += 1
                    continue

                # Load the safetensors tensor (mmap'd, not in physical RAM)
                weight_data = sf.get_tensor(key)

                # Find the target parameter in the model
                mod_path, attr_name = param_lookup[param_name]

                if mod_path:
                    # Navigate to the module
                    module = model
                    for part in mod_path.split("."):
                        module = getattr(module, part)
                else:
                    module = model

                # Get the target parameter
                target_param = getattr(module, attr_name)

                # Check if it's a Params4bit (bitsandbytes quantized layer)
                if isinstance(target_param, bnb.nn.Params4bit):
                    # CRITICAL: Create Params4bit on CPU FIRST, then move to GPU.
                    # Creating Params4bit directly on GPU keeps data as bf16 (no
                    # memory savings).  CPU creation packs into uint8 (4-bit),
                    # then moving to GPU preserves the packed format.
                    # This is a bitsandbytes 0.49.2 behavior on RTX 5060.
                    new_param = bnb.nn.Params4bit(
                        data=weight_data,  # Keep on CPU!
                        requires_grad=False,
                        quant_type="nf4",
                        compress_statistics=True,
                    )
                    # Now move the packed quantized param to GPU
                    if use_cuda:
                        new_param = new_param.to(target_device)
                    setattr(module, attr_name, new_param)
                else:
                    # Regular parameter (embedding, norm, bias, lm_head)
                    # Use target device placement — embed/lm_head stay bf16.
                    # For large params on GPU-poor systems, could offload to CPU.
                    if target_param.device.type == "meta":
                        # First time: create real parameter directly on device
                        if use_cuda:
                            real_param = nn.Parameter(
                                weight_data.to(target_device, dtype=target_param.dtype)
                            )
                        else:
                            real_param = nn.Parameter(weight_data)
                        setattr(module, attr_name, real_param)
                    else:
                        # Already has real storage: just copy data
                        target_param.data.copy_(
                            weight_data.to(target_device, dtype=target_param.dtype)
                        )

                assigned += 1

                # Free the CPU tensor immediately
                del weight_data

                # Garbage collect periodically
                if assigned % batch_size == 0:
                    gc.collect()
                    if use_cuda:
                        torch.cuda.empty_cache()

                if verbose and assigned % 100 == 0:
                    vram = torch.cuda.memory_allocated(0) / (1024**3) if use_cuda else 0
                    print(f"      {assigned} tensors, VRAM: {vram:.2f} GB")

        gc.collect()
        if use_cuda:
            torch.cuda.empty_cache()

    # ---- Step 3.5: Replace meta buffers ----
    # Buffers like rotary_emb.inv_freq are still on meta device.
    # Compute correct values and replace with real tensors.
    _realize_buffers(model, target_device, verbose=verbose)

    if verbose:
        vram_final = torch.cuda.memory_allocated(0) / (1024**3) if use_cuda else 0
        print(f"  Assigned {assigned} tensors, skipped {skipped}")
        print(f"  Final VRAM: {vram_final:.2f} GB")
        _print_device_distribution(model)

    # ---- Step 4: Load tokenizer ----
    if verbose:
        print("  Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    return model, tokenizer


def _replace_linear_with_bnb_4bit(module, verbose=False):
    """Recursively replace nn.Linear layers with bnb.nn.Linear4bit.

    Called after meta-device loading (without quantization_config) so the
    model gets 4-bit quantized layers without bitsandbytes ever touching
    from_pretrained — avoiding the segfault on RTX 5060 + Python 3.13.

    Uses a "collect first, replace later" strategy with GC disabled to
    prevent premature cleanup of meta-device parameters from triggering
    a segfault on RTX 5060 (Blackwell) + Python 3.13 + Windows.
    """
    # Phase 1: collect all (parent, child_name, child_module) tuples
    def _collect(parent, prefix=""):
        items = []
        for name, child in list(parent.named_children()):
            full = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.Linear):
                # Keep lm_head in full precision — Params4bit dtype
                # breaks resize_token_embeddings() and degrades output
                # quality for generation tasks.
                if name == "lm_head":
                    continue
                items.append((parent, name, child, full))
            elif len(list(child.children())) > 0:
                items.extend(_collect(child, full))
        return items

    all_linears = _collect(module)
    if verbose:
        print(f"  Found {len(all_linears)} Linear layers to replace")

    # Phase 2: replace one at a time with GC disabled.
    # Disabling GC prevents the cyclic garbage collector from freeing
    # orphaned meta-device nn.Linear parameters mid-loop, which can
    # segfault on this hardware/driver combination.
    gc.disable()
    try:
        for i, (parent, name, old_child, full_path) in enumerate(all_linears):
            # Create Linear4bit inside meta-device context so the
            # initial random weight is a meta tensor (no real RAM).
            # The weight-loading loop later replaces these meta
            # Params4bit params with real quantized ones from safetensors.
            with torch.device("meta"):
                new_layer = bnb.nn.Linear4bit(
                    input_features=old_child.in_features,
                    output_features=old_child.out_features,
                    bias=old_child.bias is not None,
                    compute_dtype=torch.bfloat16,
                    compress_statistics=True,
                    quant_type="nf4",
                )
            setattr(parent, name, new_layer)

            # Explicitly break references to the old meta module so any
            # deferred cleanup happens under our control.
            del old_child

            if verbose and (i + 1) % 50 == 0:
                print(f"    Replaced {i + 1}/{len(all_linears)}...")
    finally:
        gc.enable()

    return len(all_linears)


def _realize_buffers(model, target_device, verbose=False):
    """Replace meta buffers (like rotary_emb.inv_freq) with real tensors.

    When model is loaded via from_pretrained(device_map='meta'), buffers
    computed during __init__ are on meta device.  We need to re-compute
    their values on the real device.

    Qwen2 stores rope_theta in config.rope_parameters dict, not as a
    module attribute, so hasattr(module, "base") returns False.
    We extract the correct thetas from the model config.
    """
    rope_theta = 10000.0
    original_theta = 10000.0
    if hasattr(model, "config"):
        cfg = model.config
        rp = getattr(cfg, "rope_parameters", None)
        if isinstance(rp, dict):
            rope_theta = rp.get("rope_theta", rope_theta)
        original_theta = getattr(cfg, "default_theta", original_theta)

    for name, buf in list(model.named_buffers()):
        if buf.device.type != "meta":
            continue
        if "." in name:
            *mod_parts, attr = name.split(".")
            module = model
            for part in mod_parts:
                module = getattr(module, part)
        else:
            module = model
            attr = name

        if "inv_freq" in attr:
            dim = buf.shape[0] * 2
            base = original_theta if "original" in attr else rope_theta
            inv_freq = 1.0 / (
                base ** (torch.arange(0, dim, 2, dtype=torch.float32, device="cpu") / dim)
            )
            setattr(module, attr, inv_freq.to(target_device))
            if verbose:
                print(f"    Re-initialized buffer: {name} (base={base}, shape={list(buf.shape)} -> {dim} dim)")
        else:
            real_buf = torch.empty(buf.shape, dtype=buf.dtype, device=target_device)
            setattr(module, attr, real_buf)
            if verbose:
                print(f"    Replaced meta buffer: {name} (shape={list(buf.shape)})")


def _print_device_distribution(model):
    """Print how parameters are distributed across devices."""
    devices = {}
    for name, param in model.named_parameters():
        d = str(param.device)
        if d not in devices:
            devices[d] = {"count": 0, "size": 0}
        devices[d]["count"] += 1
        devices[d]["size"] += param.numel() * param.element_size()

    for dev, info in sorted(devices.items()):
        print(f"    {dev}: {info['count']} params, {info['size']/1e9:.2f} GB")


def load_lora_adapter(model, adapter_path: str, verbose: bool = True):
    """Load LoRA adapter onto an already-loaded model.

    Handles vocabulary-size mismatches between the adapter's saved
    embed_tokens/lm_head and the current base model by temporarily
    resizing the model to match before loading.
    """
    from peft import PeftModel

    if not os.path.isdir(adapter_path):
        raise FileNotFoundError(f"Adapter not found: {adapter_path}")

    if verbose:
        print(f"Loading LoRA adapter from: {adapter_path}")

    # ── Detect adapter vocab size ──────────────────────────────
    adapter_sf = os.path.join(adapter_path, "adapter_model.safetensors")
    adapter_vocab_size = None
    with safe_open(adapter_sf, framework="pt", device="cpu") as sf_:
        for key in sf_.keys():
            if key.endswith("embed_tokens.weight"):
                adapter_vocab_size = sf_.get_tensor(key).shape[0]
                break

    current_vocab = model.get_input_embeddings().weight.shape[0]
    _resized_back = None
    if adapter_vocab_size and adapter_vocab_size != current_vocab:
        if verbose:
            print(f"  Vocab mismatch: model={current_vocab} adapter={adapter_vocab_size}")
            print(f"  Resizing model vocab {current_vocab} → {adapter_vocab_size} …")
        model.resize_token_embeddings(adapter_vocab_size)
        _resized_back = current_vocab  # restore after loading

    # ── Load adapter (filter embed/lm_head to save VRAM) ──────
    import peft.utils.save_and_load as _peft_io
    _orig_load = _peft_io.load_peft_weights

    def _filtered_load(model_id, device=None, **kwargs):
        weights = _orig_load(model_id, device=device, **kwargs)
        skipped = [k for k in weights if 'embed_tokens' in k or 'lm_head' in k]
        for k in skipped:
            del weights[k]
        if verbose and skipped:
            print(f"  Skipped {len(skipped)} embed/lm_head tensors (saved VRAM)")
        return weights

    _peft_io.load_peft_weights = _filtered_load
    try:
        model = PeftModel.from_pretrained(model, adapter_path)
    finally:
        _peft_io.load_peft_weights = _orig_load

    # ── Restore original vocab size ───────────────────────────
    if _resized_back:
        if verbose:
            print(f"  Restoring vocab {adapter_vocab_size} → {_resized_back} …")
        model.resize_token_embeddings(_resized_back)

    model.eval()
    return model


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Manual 4-bit model loader")
    parser.add_argument("--model", default="models/qwen2.5-7b")
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    model, tokenizer = manual_load_4bit(args.model)

    if args.adapter:
        model = load_lora_adapter(model, args.adapter)

    if args.test:
        print("\n--- Quick Generation Test ---")
        prompt = "Hello, what is 2+2?"
        inputs = tokenizer(prompt, return_tensors="pt")
        # Find the model's primary device
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = "cpu"
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=20)
        print(tokenizer.decode(outputs[0], skip_special_tokens=True))

    print("\nModel loaded successfully!")
