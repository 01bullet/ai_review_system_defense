"""
Local LLM Reviewer with structured query defense.

Replaces the external API-based reviewer (DeepSeek Chat) with a
locally downloaded model that has been structured-instruction-tuned
to resist prompt injection attacks.

The model only follows review instructions in the [INST] portion
of the structured query, ignoring any instructions hidden in the
paper data ([DATA] portion).

Usage:
    reviewer = LocalLLMReviewer()
    reviewer.load(adapter_path="models/struq/struq_lora_adapter")

    result = reviewer.review(paper_text)
    # result = {"novelty": 7, "soundness": 8, ..., "overall": 7}
"""

from __future__ import annotations

# ---- Fix CUDA segfault on Blackwell GPUs (RTX 5060) ----
# Must be set BEFORE torch import.
import os as _os
_os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
_os.environ["SAFETENSORS_FAST_GPU"] = "1"
_os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"
del _os

import json
import os
import re
import sys
from pathlib import Path
from typing import Optional, Dict, List

import torch

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from struq_defense.config import (
    BASE_MODEL,
    LOCAL_MODEL_PATH,
    SPECIAL_TOKENS,
    LORA_OUTPUT,
    MERGED_OUTPUT,
    REVIEW_PROMPT,
    MAX_SEQ_LENGTH,
    DEVICE,
)
from struq_defense.frontend import SecureFrontend

# Try importing V2 config (optional)
try:
    from struq_defense.config_v2 import (
        BASE_MODEL as BASE_MODEL_V2,
        REVIEW_PROMPT as REVIEW_PROMPT_V2,
        LORA_OUTPUT as LORA_OUTPUT_V2,
        MERGED_OUTPUT as MERGED_OUTPUT_V2,
    )
    _V2_AVAILABLE = True
except ImportError:
    _V2_AVAILABLE = False


class LocalLLMReviewer:
    """Local LLM reviewer defended by structured queries.

    Loads a structured-instruction-tuned model (base + LoRA adapter
    or merged weights) and uses the SecureFrontend to encode queries.

    The defense mechanism:
    1. Review instructions and paper content are separated into [INST]
       and [DATA] channels.
    2. Paper content is recursively filtered to remove any special
       token strings that could be used for Completion attacks.
    3. The model was trained to follow instructions ONLY in the [INST]
       channel, ignoring anything in the [DATA] channel.
    """

    def __init__(self, frontend: Optional[SecureFrontend] = None):
        self.frontend = frontend or SecureFrontend()
        self.model = None
        self.tokenizer = None
        self._loaded = False

    # ---- Model loading ----

    def set_model(self, model, tokenizer, adapter_path: str | None = None):
        """Set a pre-loaded model and tokenizer (from manual_model_loader).

        Use this when from_pretrained() segfaults (Python 3.13 + Windows +
        limited RAM).  Caller is responsible for loading the model via
        manual_model_loader and optionally adding special tokens / LoRA.

        Args:
            model: The loaded model (already on the correct device).
            tokenizer: The loaded tokenizer.
            adapter_path: Path to adapter directory (for version detection).
        """
        self.model = model
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.eval()
        self._loaded = True

        # Version detection
        self._v2_model = False
        self._v2a_model = False
        self._v3_model = False
        if adapter_path:
            config_file = os.path.join(adapter_path, "struq_config.json")
            if os.path.exists(config_file):
                try:
                    with open(config_file, "r") as f:
                        cfg = json.load(f)
                    version = cfg.get("version", "")
                    if version == "v2":
                        self._v2_model = True
                    elif version == "v2_a":
                        self._v2a_model = True
                    elif version == "v3":
                        self._v3_model = True
                except Exception:
                    pass

    def load(
        self,
        adapter_path: str | None = None,
        merged_path: str | None = None,
        base_model: str = BASE_MODEL,
        use_merged: bool = False,
        verbose: bool = True,
    ):
        """Load the defended model.

        Two loading modes:
        1. Base model + LoRA adapter (smaller disk footprint)
        2. Merged model (convenient, larger disk footprint)

        Args:
            adapter_path: Path to LoRA adapter directory.
            merged_path: Path to merged model directory.
            base_model: HuggingFace base model ID.
            use_merged: If True, load from merged_path.
            verbose: Print loading progress.
        """
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

        if use_merged:
            load_path = merged_path or MERGED_OUTPUT
            if verbose:
                print(f"Loading merged model from {load_path}...")
            use_cuda = torch.cuda.is_available()
            if use_cuda:
                total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                if verbose:
                    print(f"  GPU: {torch.cuda.get_device_name(0)} ({total_gb:.1f}GB VRAM)")
                has_stale_quant = self._check_stale_quant_state(load_path)
                if has_stale_quant:
                    if verbose:
                        print("  Detected stale quant_state, loading without bnb...")
                    self.model = AutoModelForCausalLM.from_pretrained(
                        load_path,
                        torch_dtype=torch.bfloat16,
                        device_map="auto",
                        trust_remote_code=True,
                    )
                else:
                    bnb_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.bfloat16,
                        bnb_4bit_use_double_quant=True,
                    )
                    self.model = AutoModelForCausalLM.from_pretrained(
                        load_path,
                        quantization_config=bnb_config,
                        device_map={"": "cuda:0"},
                        trust_remote_code=True,
                    )
            else:
                if verbose:
                    print("  No GPU — loading merged model on CPU")
                self.model = AutoModelForCausalLM.from_pretrained(
                    load_path,
                    torch_dtype=torch.float32,
                    device_map={"": "cpu"},
                    trust_remote_code=True,
                )
            self.tokenizer = AutoTokenizer.from_pretrained(
                load_path,
                trust_remote_code=True,
            )
        else:
            adapter = adapter_path or LORA_OUTPUT
            # Resolve base model path (local > HF Hub)
            local_path = LOCAL_MODEL_PATH or os.environ.get("STRUQ_LOCAL_MODEL", "")
            if local_path and os.path.isdir(local_path):
                model_path = local_path
            else:
                # Check if we should auto-download to the default local path
                default_local = os.path.join(
                    os.path.dirname(os.path.dirname(__file__)),
                    "models", "qwen2.5-7b"
                )
                from ensure_model import ensure_base_model
                model_path = str(ensure_base_model(default_local, verbose=verbose))
            if verbose:
                print(f"Loading base model: {model_path}")
                print(f"Loading LoRA adapter: {adapter}")

            use_cuda = torch.cuda.is_available()
            if use_cuda:
                # Pre-cleaning: WDDM driver can fragment GPU memory across
                # processes.  Force a cache flush and pre-allocate to coalesce.
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                if verbose:
                    print(f"  GPU: {torch.cuda.get_device_name(0)} ({total_vram:.1f}GB VRAM)")
                # Force GPU memory coalescing: allocate then free a moderate
                # tensor so the WDDM driver reclaims fragmented blocks.
                try:
                    _warm = torch.zeros(1, device="cuda")
                    del _warm
                    torch.cuda.empty_cache()
                except Exception:
                    pass

                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                )
                # Limit CPU RAM during loading: 14.3GB safetensors mmap can
                # OOM the Windows page file.  max_memory tells accelerate to
                # load tensors directly to GPU, keeping CPU RAM usage low.
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    quantization_config=bnb_config,
                    device_map="auto",
                    max_memory={0: f"{int(total_vram * 0.9)}GB", "cpu": "2GB"},
                    trust_remote_code=True,
                )
            else:
                if verbose:
                    print("  No GPU — loading on CPU (no quantization, ~14GB RAM)")
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float32,
                    device_map={"": "cpu"},
                    trust_remote_code=True,
                )
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
            )

            # Add special tokens (matching training setup)
            if os.path.exists(os.path.join(adapter, "struq_config.json")):
                self.tokenizer.add_tokens(SPECIAL_TOKENS)
                self.model.resize_token_embeddings(len(self.tokenizer))
                # CRITICAL: Initialize new token embeddings from similar tokens.
                # Training did this before LoRA was applied; without it the LoRA
                # adapter receives random embeddings for [MARK]/[INST]/[DATA] etc.
                # and produces garbage output.
                self.frontend.initialize_special_embeddings(self.model, self.tokenizer)

            # Load LoRA adapter — with embed/lm_head kept on CPU to save VRAM.
            # The adapter contains full embed_tokens + lm_head weights (expanded
            # vocab after training).  Since we already resized + initialized
            # special token embeddings above (identical to training init), those
            # layers are identical to what the adapter provides.  We strip them
            # from the adapter and only load actual LoRA weights (~80 MB), avoiding
            # the ~2 GB VRAM hit from duplicating embed/lm_head on GPU.
            from peft import PeftModel
            import peft.utils.save_and_load as _peft_io
            _orig_load_peft_weights = _peft_io.load_peft_weights

            def _load_peft_lora_only(model_id, device=None, **kwargs):
                weights = _orig_load_peft_weights(model_id, device=device, **kwargs)
                skipped = [k for k in weights if 'embed_tokens' in k or 'lm_head' in k]
                for k in skipped:
                    del weights[k]
                if verbose and skipped:
                    print(f"  LoRA-only load: {len(skipped)} embed/lm_head tensors on CPU "
                          f"(saved ~{sum(w.numel()*2/1e9 for w in weights.values() if w is not None):.1f} GB VRAM)")
                return weights

            _peft_io.load_peft_weights = _load_peft_lora_only
            try:
                self.model = PeftModel.from_pretrained(self.model, adapter)
            finally:
                _peft_io.load_peft_weights = _orig_load_peft_weights

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model.eval()
        self._loaded = True

        # Model version detection: check struq_config.json for version
        self._v2_model = False
        self._v2a_model = False
        if use_merged:
            config_dir = merged_path or MERGED_OUTPUT
        else:
            config_dir = adapter_path or LORA_OUTPUT
        config_file = os.path.join(config_dir, "struq_config.json")
        if os.path.exists(config_file):
            try:
                with open(config_file, "r") as f:
                    cfg = json.load(f)
                version = cfg.get("version", "")
                if version == "v2":
                    self._v2_model = True
                elif version == "v2_a":
                    self._v2a_model = True
            except Exception:
                pass

        if verbose:
            version_info = ""
            if self._v2_model:
                version_info = " (V2 detected)"
            elif self._v2a_model:
                version_info = " (V2_A detected)"
            print(f"Model loaded successfully.{version_info}")

    def load_from_pretrained(
        self,
        model_path: str,
        verbose: bool = True,
    ):
        """Load a pre-trained defended model from a directory.

        Convenience wrapper around load(merged_path=...).

        Args:
            model_path: Path to model directory.
            verbose: Print loading progress.
        """
        return self.load(merged_path=model_path, use_merged=True, verbose=verbose)

    # ---- Review ----

    def review(
        self,
        paper_text: str,
        max_new_tokens: int = 1024,
        temperature: float = 0.1,
        **generation_kwargs,
    ) -> Dict:
        """Review a paper using the defended local model.

        Args:
            paper_text: Extracted text from the paper (untrusted).
            max_new_tokens: Max tokens for the review output.
            temperature: Generation temperature (low = deterministic).
            **generation_kwargs: Passed to model.generate().

        Returns:
            Dict with review results (novelty, soundness, presentation,
            overall, decision, review), plus defense metadata.
        """
        if not self._loaded:
            return {
                "error": "Model not loaded. Call reviewer.load() first.",
                "novelty": 0, "soundness": 0, "presentation": 0,
                "overall": 0, "decision": "Error", "review": "",
            }

        # Adapt sequence length to available VRAM
        safe_max_length = MAX_SEQ_LENGTH
        if torch.cuda.is_available():
            free_vram = (torch.cuda.get_device_properties(0).total_memory
                         - torch.cuda.memory_allocated()
                         - torch.cuda.memory_reserved()) / (1024**3)
            if free_vram < 2.0:
                safe_max_length = min(MAX_SEQ_LENGTH, 2048)

        # Encode structured query (filters special tokens from paper)
        # IMPORTANT: Truncate paper data so the structural markers ([RESP][COLN])
        # at the END are never truncated. Otherwise the model generates garbage.
        # Check for prompt override first (e.g. v3a_baseline no-defense mode)
        if getattr(self, '_prompt_override', None) is not None:
            prompt = self._prompt_override
        else:
            prompt = REVIEW_PROMPT
            if _V2_AVAILABLE and getattr(self, '_v2_model', False):
                prompt = REVIEW_PROMPT_V2
            elif getattr(self, '_v2a_model', False) or getattr(self, '_v3_model', False):
                # V2_A and V3 use the same enhanced prompt
                try:
                    from struq_defense.config_v2_a import REVIEW_PROMPT as V2A_PROMPT
                    prompt = V2A_PROMPT
                except ImportError:
                    pass

        prompt_len = len(self.tokenizer.encode(prompt, add_special_tokens=False))
        marker_len = len(self.tokenizer.encode("[MARK][DATA][COLN]\n\n[MARK][RESP][COLN]\n", add_special_tokens=False))
        max_data_tokens = safe_max_length - prompt_len - marker_len - 100
        # Use V2 filter for V2 models; V1 filter for V2_A and others
        if _V2_AVAILABLE and getattr(self, '_v2_model', False) and hasattr(self.frontend, 'filter_data_v2'):
            paper_text = self.frontend.filter_data_v2(paper_text)
        else:
            paper_text = self.frontend.filter_data(paper_text)
        paper_tokens = self.tokenizer.encode(paper_text, add_special_tokens=False)
        if len(paper_tokens) > max_data_tokens:
            paper_tokens = paper_tokens[:max_data_tokens]
            paper_text = self.tokenizer.decode(paper_tokens)
        encoded_query = self.frontend.encode(prompt, paper_text)

        # Tokenize
        inputs = self.tokenizer(
            encoded_query,
            return_tensors="pt",
            truncation=True,
            max_length=safe_max_length,
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # Generate
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=(temperature > 0),
                top_p=0.95,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                **generation_kwargs,
            )

        # Decode
        full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        response = self.frontend.decode(full_output)

        # Parse JSON review
        review_data = self._parse_review_json(response)

        # Add defense metadata
        review_data["_defense"] = {
            "method": "struq_structured_query",
            "model": "local_defended",
            "query_format": "mark_inst_coln",
            "response_raw": response[:500],
        }

        return review_data

    def review_batch(
        self,
        papers: List[str],
        **generation_kwargs,
    ) -> List[Dict]:
        """Review multiple papers.

        Args:
            papers: List of paper texts.
            **generation_kwargs: Passed to review().

        Returns:
            List of review dicts.
        """
        return [self.review(paper, **generation_kwargs) for paper in papers]

    # ---- JSON parsing ----

    def _parse_review_json(self, response: str) -> Dict:
        """Extract review JSON from model response.

        Handles multiple output formats:
        1. Proper JSON: {"novelty": 7, ...}
        2. Key-value: novelty: 7, soundness: 8, ...
        3. YAML-like with colons

        Args:
            response: Raw model response text.

        Returns:
            Parsed review dict, or default error dict.
        """
        defaults = {
            "novelty": 0,
            "soundness": 0,
            "presentation": 0,
            "overall": 0,
            "decision": "Parse Error",
            "review": response.strip(),
        }

        # Strategy 1: Bracket-counting JSON extraction.
        # Handles nested braces inside review text, markdown fences,
        # and leading noise (e.g. ':\\n' from [COLN] token).
        text = response.strip()
        text = re.sub(r'```(?:json)?\s*', '', text).replace('```', '').strip()

        brace_idx = text.find('{')
        if brace_idx >= 0:
            text = text[brace_idx:]
            depth = 0
            in_string = False
            escape_next = False
            end_idx = -1
            for i, ch in enumerate(text):
                if escape_next:
                    escape_next = False
                    continue
                if ch == '\\':
                    escape_next = True
                    continue
                if ch == '"' and not escape_next:
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                if ch == '{':
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0:
                        end_idx = i + 1
                        break

            # Check if the extracted JSON has review keys
            json_str = text[:end_idx] if end_idx > 0 else text
            has_keys = any(k in json_str for k in ['"novelty"', '"overall"', '"decision"'])
            if end_idx > 0 and has_keys:
                try:
                    parsed = json.loads(json_str)
                    result = {k.lower(): v for k, v in parsed.items()}
                    for key in ["novelty", "soundness", "presentation", "overall"]:
                        if key not in result:
                            result[key] = defaults[key]
                    if "decision" not in result:
                        result["decision"] = "Accept" if result.get("overall", 0) >= 7 else "Reject"
                    if "review" not in result:
                        result["review"] = response.strip()
                    return result
                except json.JSONDecodeError:
                    pass

        # Fallback: simple regex for flat JSON (no nested braces)
        json_match = re.search(r'\{[^{}]*"novelty"[^{}]*\}', text, re.DOTALL)
        if not json_match:
            json_match = re.search(r'\{[^{}]*"overall"[^{}]*\}', text, re.DOTALL)
        if not json_match:
            json_match = re.search(r'\{[^{}]*"decision"[^{}]*\}', text, re.DOTALL)

        if json_match:
            try:
                parsed = json.loads(json_match.group())
                result = {k.lower(): v for k, v in parsed.items()}
                for key in ["novelty", "soundness", "presentation", "overall"]:
                    if key not in result:
                        result[key] = defaults[key]
                if "decision" not in result:
                    result["decision"] = "Accept" if result.get("overall", 0) >= 7 else "Reject"
                if "review" not in result:
                    result["review"] = response.strip()
                return result
            except json.JSONDecodeError:
                pass

        # Strategy 2: Extract key: value pairs (handles "novelty: 7" format)
        score_patterns = {
            "novelty": r'(?:novelty|Novelty)\s*[:=]\s*(\d+)',
            "soundness": r'(?:soundness|Soundness)\s*[:=]\s*(\d+)',
            "presentation": r'(?:presentation|Presentation)\s*[:=]\s*(\d+)',
            "overall": r'(?:overall|Overall)\s*[:=]\s*(\d+)',
        }
        decision_match = re.search(
            r'(?:decision|Decision)\s*[:=]\s*(Accept|Reject|accept|reject)',
            response
        )

        scores = {}
        for key, pattern in score_patterns.items():
            m = re.search(pattern, response)
            if m:
                scores[key] = int(m.group(1))

        if scores:
            # Merge with defaults
            result = {**defaults, **scores}
            if decision_match:
                result["decision"] = decision_match.group(1).capitalize()
            else:
                overall = scores.get("overall", 0)
                result["decision"] = "Accept" if overall >= 7 else "Reject"
            # Use text after the scores as review
            result["review"] = response.strip()
            return result

        # Strategy 3: Extract loose "score: X" or "score X/10" patterns
        overall_patterns = [
            r'(?:overall\s+)?score\s*[:=]\s*(\d+)',
            r'(?:overall\s+)?score\s*[:=]\s*(\d+)\s*/\s*10',
            r'(?:overall\s+)?rating\s*[:=]\s*(\d+)',
            r'(?:my\s+)?score\s*(?:is\s*)?[:=]?\s*(\d+)',
            r'(?:give|gave|giving)\s+(?:it|this|a)\s+(?:a\s+)?score\s+(?:of\s+)?(\d+)',
            r'(?:overall|Overall)(?:\s+score)?(?:\s+is)?[:=]?\s*(\d+)',
        ]
        overall_score = 0
        for pat in overall_patterns:
            m = re.search(pat, response)
            if m:
                overall_score = int(m.group(1))
                break

        if overall_score > 0:
            scores["overall"] = overall_score
            # Estimate missing dimension scores from overall
            if "novelty" not in scores:
                scores["novelty"] = overall_score
            if "soundness" not in scores:
                scores["soundness"] = overall_score
            if "presentation" not in scores:
                scores["presentation"] = overall_score

        if scores:
            result = {**defaults, **scores}
            result["decision"] = "Accept" if scores.get("overall", 0) >= 7 else "Reject"
            result["review"] = response.strip()
            return result

        # Strategy 4: Bracket-counting extraction for any JSON-like {...} block
        brace_idx = response.find('{')
        if brace_idx >= 0:
            text = response[brace_idx:]
            depth = 0
            in_string = False
            escape_next = False
            end_idx = -1
            for i, ch in enumerate(text):
                if escape_next:
                    escape_next = False
                    continue
                if ch == '\\':
                    escape_next = True
                    continue
                if ch == '"':
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                if ch == '{':
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0:
                        end_idx = i + 1
                        break
            if end_idx > 0:
                try:
                    parsed = json.loads(text[:end_idx])
                    result = {k.lower(): v for k, v in parsed.items()}
                    for key in ["novelty", "soundness", "presentation", "overall"]:
                        if key not in result:
                            result[key] = defaults[key]
                    if "decision" not in result:
                        result["decision"] = "Accept" if result.get("overall", 0) >= 7 else "Reject"
                    if "review" not in result:
                        result["review"] = response.strip()
                    return result
                except json.JSONDecodeError:
                    pass

        return defaults

    # ---- Utility ----

    @staticmethod
    def _check_stale_quant_state(load_path: str) -> bool:
        """Check if merged model safetensors contain stale quant_state keys.

        When a 4-bit loaded model is merged and saved, the safetensors may
        retain quant_state. Attempting to re-quantize with bnb 4-bit on load
        causes shape mismatch errors.
        """
        import json
        from pathlib import Path

        index_path = Path(load_path) / "model.safetensors.index.json"
        if not index_path.exists():
            return False
        with open(index_path) as f:
            index = json.load(f)
        weight_map = index.get("weight_map", {})
        # Check if any key in the first shard has quant_state
        for key in weight_map:
            if "quant_state" in key or "nested_absmax" in key or "quant_map" in key:
                return True
        return False

    def is_loaded(self) -> bool:
        """Check if model is loaded and ready."""
        return self._loaded and self.model is not None

    def compare_with_api(
        self,
        paper_text: str,
        api_review: Dict,
    ) -> Dict:
        """Compare local model review with API review.

        Useful for evaluating how well the local model matches
        the trusted API reviewer on clean papers.

        Args:
            paper_text: Paper content.
            api_review: Review from the trusted API.

        Returns:
            Dict with comparison metrics.
        """
        local = self.review(paper_text)

        score_diff = {}
        for key in ["novelty", "soundness", "presentation", "overall"]:
            score_diff[key] = abs(local.get(key, 0) - api_review.get(key, 0))

        score_diff["mean_abs_error"] = sum(score_diff.values()) / len(score_diff)

        return {
            "local_review": local,
            "api_review": api_review,
            "score_diffs": score_diff,
            "decision_match": local.get("decision") == api_review.get("decision"),
        }
