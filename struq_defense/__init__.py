"""
StruQ Defense — Structured Query Defense for AI Scientist Review System.

============================================================
FEASIBILITY ANALYSIS
============================================================

StruQ (USENIX Security 2025, UC Berkeley) proposes separating LLM
prompts (instructions) from data (user content) via structured queries.
The core insight: prompt injection is a classic "control mixed with data"
vulnerability, solved by separating them into two channels.

APPLIED TO THIS PROJECT:
1. FRONT-END: Separate review instructions from paper content using
   special reserved tokens ([MARK], [INST], [DATA], [RESP], [COLN]).
   Filter these tokens from paper data to prevent Completion attacks.

2. LOCAL MODEL: Replace external DeepSeek API with a downloaded base
   model (Qwen2.5-7B / Llama-3.1-8B), structured-instruction-tuned to
   follow instructions ONLY in the prompt portion, ignoring injections
   hidden in the paper data.

3. TRAINING DATA CONSTRUCTION — using EXISTING components:
   - AttackPayloadGenerator → generates novel attack payloads
   - HidingSelector (22+7 techniques) → wraps payloads in LaTeX
   - ATTACK_STRATEGIES (17 rule-based) → supplementary diversity
   - perform_review() (API) → generates "clean review" targets

   Dataset composition (following StruQ Section 4.4):
   - 50% Clean: paper_text + clean review (the ground truth)
   - 25% Naive attack: Generator payload → injected paper + SAME clean review
   - 25% Completion attack: fake delimiters + payload → injected paper + SAME review

   Key insight: ALL variants use the SAME clean review as target.
   This teaches the model: "no matter what's in the paper data,
   output the honest review based only on the real prompt instructions."

4. TRAINING: QLoRA fine-tuning (4-bit, LoRA adapters) — feasible on
   a single 16GB GPU. New special tokens' embeddings initialized from
   semantically similar existing tokens (StruQ key finding).

5. DEFENSE EVALUATION: Test the fine-tuned model against Generator
   attacks, rule-based attacks, and optimization attacks (TAP-style).

ARCHITECTURE:
    ┌──────────────────────┐    ┌──────────────────────────────┐
    │   Secure Front-End    │    │   Local LLM Reviewer          │
    │                      │    │                              │
    │  [MARK][INST][COLN]  │───▶│  Qwen2.5-7B / Llama-3.1-8B  │
    │  review instructions  │    │  + QLoRA adapters            │
    │  [MARK][DATA][COLN]  │    │                              │
    │  paper content        │    │  Only follows [INST],       │
    │  [MARK][RESP][COLN]  │    │  ignores injections in [DATA]│
    │                      │    │                              │
    │  Recursive filter:   │    │  Output: review JSON          │
    │  [MARK],[INST],etc.  │    └──────────────────────────────┘
    └──────────────────────┘

    ┌──────────────────────────────────────────────────────────┐
    │              Training Data Generation Pipeline            │
    │                                                          │
    │  Clean Papers ──▶ API Reviewer ──▶ clean_review (target) │
    │       │                                                  │
    │       ├──▶ 50% Clean: (paper, clean_review)              │
    │       ├──▶ 25% Naive: Generator→HidingSelector→inject   │
    │       │         → (attacked_paper, clean_review)          │
    │       └──▶ 25% Completion: fake delimiters + payload     │
    │                → (attacked_paper, clean_review)           │
    └──────────────────────────────────────────────────────────┘

============================================================
USAGE:
    # 1. Build training dataset (requires API for clean reviews)
    python -m struq_defense.run build-dataset --papers example_papers/

    # 2. Train the local defended model
    python -m struq_defense.run train --dataset data/struq_dataset.json

    # 3. Run review with the defended model
    python -m struq_defense.run review --paper example_papers/xxx/xxx.pdf

    # 4. Evaluate defense against attacks
    python -m struq_defense.run evaluate --papers example_papers/
"""

__version__ = "0.1.0"
