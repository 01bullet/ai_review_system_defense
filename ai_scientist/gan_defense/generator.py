"""
Generator model for GAN-based adversarial attack generation.

A T5-small model fine-tuned to generate novel prompt injection payloads.
The generator takes a context + style + target specification and produces
variant injection text that attempts to bypass rule-based defenses.

Training uses a two-phase approach:
1. Supervised fine-tuning on known attack payloads
2. REINFORCE policy gradient with discriminator feedback
"""

import os
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple

from ai_scientist.gan_defense.config import (
    GENERATOR_MODEL,
    GENERATOR_DIR,
    G_MAX_LENGTH,
    G_ADVERSARIAL_LR,
    DEVICE,
)


class AttackPayloadGenerator(nn.Module):
    """T5-small based generator for adversarial injection payloads.

    Generates the TEXT of the injection (English payload), not the
    LaTeX wrapping commands. The HidingSelector handles LaTeX wrapping.
    """

    def __init__(self, model_name: str = GENERATOR_MODEL):
        super().__init__()
        from transformers import T5ForConditionalGeneration

        try:
            self.model = T5ForConditionalGeneration.from_pretrained(
                model_name, local_files_only=True
            )
        except Exception:
            self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.model_name = model_name
        self.max_length = G_MAX_LENGTH

    def forward(self, input_ids, attention_mask, labels=None):
        """Standard T5 forward pass.

        Args:
            input_ids: (batch, seq_len) encoder input.
            attention_mask: (batch, seq_len) attention mask.
            labels: (batch, seq_len) decoder targets for teacher forcing.

        Returns:
            T5 model output with loss when labels provided.
        """
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

    def generate(
        self,
        input_text: str,
        tokenizer=None,
        temperature: float = 0.8,
        do_sample: bool = True,
        num_return_sequences: int = 4,
    ) -> List[str]:
        """Generate adversarial payload text.

        Args:
            input_text: Prompt like "generate injection: style=X target=Y context=Z"
            tokenizer: T5 tokenizer (created lazily if None).
            temperature: Sampling temperature (higher = more diverse).
            do_sample: Use nucleus sampling (True) vs greedy (False).
            num_return_sequences: Number of variant payloads to generate.

        Returns:
            List of generated payload strings.
        """
        if tokenizer is None:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        self.eval()

        encoded = tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )
        input_ids = encoded["input_ids"].to(DEVICE)
        attention_mask = encoded["attention_mask"].to(DEVICE)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=self.max_length,
                num_return_sequences=min(num_return_sequences, 4),
                do_sample=do_sample,
                temperature=temperature,
                top_p=0.95,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        payloads = [
            tokenizer.decode(out, skip_special_tokens=True)
            for out in outputs
        ]

        return [p for p in payloads if len(p.strip()) > 10]

    def generate_single(self, input_text: str, tokenizer=None) -> str:
        """Generate a single payload string.

        Args:
            input_text: Input prompt.
            tokenizer: T5 tokenizer.

        Returns:
            Generated payload string, or empty string on failure.
        """
        payloads = self.generate(
            input_text, tokenizer, do_sample=True, num_return_sequences=1
        )
        return payloads[0] if payloads else ""

    def batch_rollout(
        self,
        input_texts: List[str],
        tokenizer=None,
        temperature: float = 0.8,
        top_p: float = 0.95,
    ) -> List[str]:
        """Generate payloads for a batch of input prompts.

        More efficient than calling generate_single() in a loop because
        it batches the tokenization and generation steps.

        Args:
            input_texts: List of input prompts.
            tokenizer: T5 tokenizer.
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.

        Returns:
            List of generated payload strings (one per input).
        """
        if not input_texts:
            return []

        if tokenizer is None:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        self.eval()

        encoded = tokenizer(
            input_texts,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )
        input_ids = encoded["input_ids"].to(DEVICE)
        attention_mask = encoded["attention_mask"].to(DEVICE)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=self.max_length,
                num_return_sequences=1,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        payloads = [
            tokenizer.decode(out, skip_special_tokens=True)
            for out in outputs
        ]
        return [p for p in payloads if len(p.strip()) > 10]

    def generate_with_style(
        self,
        context_text: str,
        style: str = "authoritative",
        target_score: int = 10,
        tokenizer=None,
    ) -> str:
        """Convenience method: generate a payload with a specific style and target.

        Args:
            context_text: Paper context (first 256 chars of the paper).
            style: One of authoritative, technical, academic, casual.
            target_score: Target Overall score (1-10).
            tokenizer: T5 tokenizer.

        Returns:
            Generated payload string.
        """
        input_text = (
            f"generate injection: style={style} "
            f"target={target_score} context={context_text[:256]}"
        )
        return self.generate_single(input_text, tokenizer)

    def save(self, path: Optional[str] = None):
        """Save generator weights."""
        save_path = path or os.path.join(GENERATOR_DIR, "generator.pt")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save({
            "model_state_dict": self.state_dict(),
            "model_name": self.model_name,
        }, save_path)

    def load(self, path: Optional[str] = None):
        """Load generator weights."""
        load_path = path or os.path.join(GENERATOR_DIR, "generator.pt")
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"No checkpoint at {load_path}")
        checkpoint = torch.load(load_path, map_location=DEVICE, weights_only=False)
        self.load_state_dict(checkpoint["model_state_dict"])


class ReinforceTrainer:
    """REINFORCE policy gradient trainer for the Generator.

    Handles the RL-based fine-tuning. The reward signal can come from:
    - Discriminator scores + sanitizer bypass (adversarial training)
    - LLM reviewer scores (RL training with direct LLM feedback)

    Supports both D-based and LLM-based reward sources.
    """

    def __init__(
        self,
        generator: AttackPayloadGenerator,
        lr: float = G_ADVERSARIAL_LR,
        entropy_bonus: float = 0.01,
        baseline_decay: float = 0.95,
        reward_source: str = "unknown",
    ):
        self.generator = generator
        self.optimizer = torch.optim.AdamW(
            generator.model.parameters(), lr=lr
        )
        self.entropy_bonus = entropy_bonus
        self.baseline_decay = baseline_decay
        self.running_baseline = 0.0  # EMA of past rewards
        self.reward_source = reward_source
        self.n_updates = 0  # track number of updates for LR scheduling

    def update(
        self,
        input_texts: List[str],
        generated_payloads: List[str],
        rewards: List[float],
        tokenizer=None,
    ) -> Dict[str, float]:
        """Single REINFORCE update step.

        Computes policy gradient loss:
          loss = -mean(log_prob(action) * (reward - baseline))
          + entropy_bonus * -entropy

        Args:
            input_texts: Input prompts that produced the payloads.
            generated_payloads: The generated payload strings.
            rewards: Reward for each payload ∈ [-1, 1].
            tokenizer: T5 tokenizer.

        Returns:
            Dict with training metrics (loss, mean_reward, entropy).
        """
        if tokenizer is None:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(self.generator.model_name)

        if not generated_payloads or not rewards:
            return {"loss": 0.0, "mean_reward": 0.0, "entropy": 0.0}

        self.generator.train()

        # Encode inputs
        input_encoded = tokenizer(
            input_texts,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )
        input_ids = input_encoded["input_ids"].to(DEVICE)
        input_attn = input_encoded["attention_mask"].to(DEVICE)

        # Encode targets
        target_encoded = tokenizer(
            generated_payloads,
            return_tensors="pt",
            truncation=True,
            max_length=self.generator.max_length,
            padding=True,
        )
        target_ids = target_encoded["input_ids"].to(DEVICE)
        target_attn = target_encoded["attention_mask"].to(DEVICE)

        # Replace padding with -100 (ignored in loss)
        target_ids[target_ids == tokenizer.pad_token_id] = -100

        # Forward pass with teacher forcing to get log probabilities
        outputs = self.generator.model(
            input_ids=input_ids,
            attention_mask=input_attn,
            labels=target_ids,
        )

        # Per-sample loss (CE with teacher forcing)
        # outputs.loss is averaged; compute per-sample
        loss_per_sample = []
        seq_len = target_ids.size(1)

        logits = outputs.logits  # (batch, seq, vocab)

        for i in range(len(generated_payloads)):
            # Compute CE loss for this sample
            shift_logits = logits[i, :-1, :]  # predict next token
            shift_labels = target_ids[i, 1:]  # skip first token

            ce = torch.nn.functional.cross_entropy(
                shift_logits, shift_labels,
                ignore_index=-100,
                reduction='mean',
            )

            # Negative log prob as policy loss
            # REINFORCE: L = -log_prob * advantage
            log_prob = -ce  # CE = -log(p), so CE = -log_prob
            loss_per_sample.append(log_prob)

        # Apply REINFORCE
        rewards_tensor = torch.tensor(rewards, device=DEVICE)

        # Update running baseline
        mean_reward = rewards_tensor.mean().item()
        self.running_baseline = (
            self.baseline_decay * self.running_baseline
            + (1 - self.baseline_decay) * mean_reward
        )

        advantages = rewards_tensor - self.running_baseline

        # Normalize advantages for stable training
        if len(advantages) > 1 and advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Policy gradient loss
        policy_loss = torch.stack([
            -lp * adv
            for lp, adv in zip(loss_per_sample, advantages)
        ]).mean()

        # Entropy bonus (encourage diversity)
        probs = torch.softmax(logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(-1).mean()
        total_loss = policy_loss + self.entropy_bonus * (-entropy)

        # Update
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.generator.model.parameters(), 1.0)
        self.optimizer.step()

        self.n_updates += 1

        return {
            "loss": total_loss.item(),
            "mean_reward": mean_reward,
            "entropy": entropy.item(),
            "baseline": self.running_baseline,
            "n_updates": self.n_updates,
        }


def create_generator(model_name: str = GENERATOR_MODEL) -> AttackPayloadGenerator:
    """Factory to create and initialize the generator.

    Args:
        model_name: HuggingFace model identifier.

    Returns:
        Initialized AttackPayloadGenerator on the configured device.
    """
    gen = AttackPayloadGenerator(model_name=model_name)
    gen.to(DEVICE)
    return gen
