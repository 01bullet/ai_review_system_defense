"""
Adversarial training loop for the GAN defense system.

Orchestrates the two-player game between Generator (G) and Discriminator (D):
- G learns to create attack payloads that bypass both D and the rule-based sanitizer
- D learns to detect increasingly sophisticated injection attacks

Training proceeds in two phases:
1. Pre-training (offline): Supervised training of D and G individually
2. Adversarial training: Alternating G and D updates with REINFORCE
"""

import os
import time
import json
import random
import torch
from typing import Dict, List, Tuple, Optional

from ai_scientist.gan_defense.config import (
    DEVICE,
    D_PRETRAIN_EPOCHS,
    D_PRETRAIN_LR,
    D_BATCH_SIZE,
    D_ADVERSARIAL_LR,
    D_OLD_DATA_FRACTION,
    G_PRETRAIN_EPOCHS,
    G_PRETRAIN_LR,
    G_SAMPLES_PER_ITER,
    D_BATCH_PER_ITER,
    ADVERSARIAL_ITERATIONS,
    DISCRIMINATOR_DIR,
    GENERATOR_DIR,
)

from ai_scientist.gan_defense.discriminator import (
    PaperDiscriminator,
    create_discriminator,
)

from ai_scientist.gan_defense.generator import (
    AttackPayloadGenerator,
    ReinforceTrainer,
    create_generator,
)

from ai_scientist.gan_defense.hiding_selector import HidingSelector
from ai_scientist.gan_defense.reward import (
    compute_rewards,
    compute_reward_stats,
    apply_validity_penalty,
)
from ai_scientist.gan_defense.data_utils import (
    load_clean_papers,
    build_discriminator_dataset,
    build_generator_dataset,
    extract_text_from_latex_fast,
    sanitizer_bypass_score,
)


class GanAdversarialTrainer:
    """Orchestrates the GAN adversarial training loop."""

    def __init__(
        self,
        discriminator: Optional[PaperDiscriminator] = None,
        generator: Optional[AttackPayloadGenerator] = None,
        papers_dir: Optional[str] = None,
    ):
        self.discriminator = discriminator or create_discriminator()
        self.generator = generator or create_generator()
        self.hiding_selector = HidingSelector(exploration_rate=0.3)

        self.rl_trainer = ReinforceTrainer(self.generator)

        # Tokenizers
        from transformers import AutoTokenizer
        self.d_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.g_tokenizer = AutoTokenizer.from_pretrained("t5-small")

        # Paper cache
        self.papers = load_clean_papers(papers_dir)
        print(f"[GAN Trainer] Loaded {len(self.papers)} clean papers for training.")

        # Metrics log
        self.metrics_history: List[Dict] = []

    # ============= Phase 1: Pre-training =============

    def pretrain_discriminator(
        self,
        epochs: int = D_PRETRAIN_EPOCHS,
        lr: float = D_PRETRAIN_LR,
        batch_size: int = D_BATCH_SIZE,
    ) -> Dict:
        """Pre-train discriminator on known clean and attacked papers.

        Args:
            epochs: Number of training epochs.
            lr: Learning rate.
            batch_size: Batch size.

        Returns:
            Dict of training metrics.
        """
        print(f"\n[GAN Trainer] Pre-training Discriminator ({epochs} epochs)...")

        texts, labels = build_discriminator_dataset(self.papers)
        if not texts:
            print("[GAN Trainer] WARNING: No training data for D.")
            return {}

        optimizer = torch.optim.AdamW(
            self.discriminator.parameters(), lr=lr
        )

        n_samples = len(texts)
        metrics = []

        for epoch in range(epochs):
            epoch_losses = []
            indices = list(range(n_samples))
            random.shuffle(indices)

            for i in range(0, n_samples, batch_size):
                batch_idx = indices[i:i + batch_size]
                batch_texts = [texts[j] for j in batch_idx]
                batch_labels = [labels[j] for j in batch_idx]

                # Tokenize
                encoded = self.d_tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding=True,
                )
                input_ids = encoded["input_ids"].to(DEVICE)
                attention_mask = encoded["attention_mask"].to(DEVICE)
                targets = torch.tensor(batch_labels, dtype=torch.float32).to(DEVICE)

                # Forward
                self.discriminator.train()
                preds = self.discriminator(input_ids, attention_mask).squeeze(-1)

                # BCE with label smoothing
                smoothed_targets = targets * (1 - D_BATCH_SIZE * 0.01) + 0.005
                loss = torch.nn.functional.binary_cross_entropy(
                    preds, smoothed_targets
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_losses.append(loss.item())

            avg_loss = sum(epoch_losses) / len(epoch_losses)
            acc = self._evaluate_d_accuracy(texts[:batch_size * 4], labels[:batch_size * 4])
            metrics.append({"epoch": epoch, "loss": avg_loss, "accuracy": acc})
            print(f"  Epoch {epoch + 1}/{epochs} — loss: {avg_loss:.4f}, acc: {acc:.3f}")

        # Save checkpoint
        self.discriminator.save()
        return {"pretrain_d_metrics": metrics}

    def pretrain_generator(
        self,
        epochs: int = G_PRETRAIN_EPOCHS,
        lr: float = G_PRETRAIN_LR,
    ) -> Dict:
        """Pre-train generator via supervised fine-tuning on known attack payloads.

        Args:
            epochs: Number of training epochs.
            lr: Learning rate.

        Returns:
            Dict of training metrics.
        """
        print(f"\n[GAN Trainer] Pre-training Generator ({epochs} epochs)...")

        examples = build_generator_dataset(self.papers)
        if not examples:
            print("[GAN Trainer] WARNING: No training data for G.")
            return {}

        optimizer = torch.optim.AdamW(
            self.generator.parameters(), lr=lr
        )

        n_samples = len(examples)
        batch_size = 4
        metrics = []

        for epoch in range(epochs):
            epoch_losses = []
            indices = list(range(n_samples))
            random.shuffle(indices)

            for i in range(0, n_samples, batch_size):
                batch_idx = indices[i:i + batch_size]
                batch_inputs = [examples[j]["input"] for j in batch_idx]
                batch_targets = [examples[j]["target"] for j in batch_idx]

                # Tokenize inputs
                input_enc = self.g_tokenizer(
                    batch_inputs,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding=True,
                )
                input_ids = input_enc["input_ids"].to(DEVICE)
                input_attn = input_enc["attention_mask"].to(DEVICE)

                # Tokenize targets
                target_enc = self.g_tokenizer(
                    batch_targets,
                    return_tensors="pt",
                    truncation=True,
                    max_length=256,
                    padding=True,
                )

                labels = target_enc["input_ids"].to(DEVICE)
                labels[labels == self.g_tokenizer.pad_token_id] = -100

                # Forward
                self.generator.train()
                outputs = self.generator.model(
                    input_ids=input_ids,
                    attention_mask=input_attn,
                    labels=labels,
                )
                loss = outputs.loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_losses.append(loss.item())

            avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0
            metrics.append({"epoch": epoch, "loss": avg_loss})
            print(f"  Epoch {epoch + 1}/{epochs} — loss: {avg_loss:.4f}")

        self.generator.save()
        return {"pretrain_g_metrics": metrics}

    # ============= Phase 2: Adversarial Training =============

    def adversarial_train(
        self,
        iterations: int = ADVERSARIAL_ITERATIONS,
        samples_per_iter: int = G_SAMPLES_PER_ITER,
        d_batch_size: int = D_BATCH_PER_ITER,
    ) -> Dict:
        """Run the adversarial training loop.

        Alternates between:
        - G: Generate attacks → compute rewards → REINFORCE update
        - D: Train on clean + hard negatives + cached attacks

        Args:
            iterations: Number of adversarial iterations.
            samples_per_iter: Number of attacks G generates per iteration.
            d_batch_size: Batch size for D update.

        Returns:
            Dict with full metrics history.
        """
        print(f"\n[GAN Trainer] Starting adversarial training ({iterations} iters)...")

        d_optimizer = torch.optim.AdamW(
            self.discriminator.parameters(), lr=D_ADVERSARIAL_LR
        )

        # Cache of old attacked texts (for D's replay buffer)
        attacked_cache: List[Tuple[str, int]] = []  # (text, label=1)
        max_cache = 500

        # Paper pool for G to attack
        paper_pool = self.papers[:20] if len(self.papers) > 20 else self.papers

        for iteration in range(iterations):
            # ===== Step 1: G generates attacks =====
            generated_latex_list = []
            generated_texts = []
            generated_payloads = []
            input_prompts = []

            for _ in range(samples_per_iter):
                paper = random.choice(paper_pool)
                context = paper["text_content"][:256]

                style = random.choice(["authoritative", "technical", "academic", "casual"])
                target = random.choice([7, 8, 9, 10])

                input_text = (
                    f"generate injection: style={style} "
                    f"target={target} context={context}"
                )

                payload = self.generator.generate_single(input_text, self.g_tokenizer)
                if not payload:
                    continue

                technique = self.hiding_selector.sample_technique()
                injection_point = self.hiding_selector.sample_injection_point()

                attacked_latex = self.hiding_selector.inject_into_latex(
                    paper["latex_content"], payload, technique, injection_point
                )
                attacked_text = extract_text_from_latex_fast(attacked_latex)

                generated_latex_list.append(attacked_latex)
                generated_texts.append(attacked_text)
                generated_payloads.append(payload)
                input_prompts.append(input_text)

            if not generated_texts:
                continue

            # ===== Step 2: D scores the generated attacks =====
            d_scores = self.discriminator.predict_batch(
                generated_texts, self.d_tokenizer
            )

            # ===== Step 3: Compute rewards for G =====
            rewards = compute_rewards(d_scores, generated_latex_list)
            rewards = apply_validity_penalty(rewards, generated_payloads)

            reward_stats = compute_reward_stats(rewards)

            # Update HidingSelector stats
            for i, (latex, reward) in enumerate(zip(generated_latex_list, rewards)):
                bypassed = sanitizer_bypass_score(latex) > 0.5
                # Track the technique used
                pass  # technique tracking simplified for now

            # ===== Step 4: Update G via REINFORCE =====
            rl_metrics = self.rl_trainer.update(
                input_prompts, generated_payloads, rewards, self.g_tokenizer
            )

            # ===== Step 5: Update D on hard negatives =====
            # Identify hard negatives (D scored low on real attacks)
            difficulties = [1.0 - s for s in d_scores]
            hard_pairs = sorted(
                zip(generated_texts, difficulties),
                key=lambda x: x[1],
                reverse=True,
            )[:max(4, d_batch_size // 2)]

            hard_texts = [t for t, _ in hard_pairs]

            # Build D batch: clean + hard negatives + cached
            d_texts = []
            d_labels = []

            # Clean samples
            n_clean = min(d_batch_size // 3, len(paper_pool))
            clean_papers = random.sample(paper_pool, n_clean)
            for p in clean_papers:
                d_texts.append(p["text_content"])
                d_labels.append(0)

            # Hard negatives (G's best attacks)
            for t in hard_texts:
                d_texts.append(t)
                d_labels.append(1)
                attacked_cache.append((t, 1))

            # Cached old attacks
            if attacked_cache:
                n_old = min(
                    int(d_batch_size * D_OLD_DATA_FRACTION),
                    len(attacked_cache),
                )
                old_samples = random.sample(attacked_cache, n_old)
                for t, l in old_samples:
                    d_texts.append(t)
                    d_labels.append(l)

            # Trim cache
            if len(attacked_cache) > max_cache:
                attacked_cache = attacked_cache[-max_cache:]

            # Train D
            d_loss_sum = 0.0
            n_d_batches = 0

            combined = list(zip(d_texts, d_labels))
            random.shuffle(combined)

            for i in range(0, len(combined), d_batch_size):
                batch = combined[i:i + d_batch_size]
                b_texts, b_labels = zip(*batch)

                encoded = self.d_tokenizer(
                    list(b_texts),
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding=True,
                )
                input_ids = encoded["input_ids"].to(DEVICE)
                attn = encoded["attention_mask"].to(DEVICE)
                targets = torch.tensor(b_labels, dtype=torch.float32).to(DEVICE)

                self.discriminator.train()
                preds = self.discriminator(input_ids, attn).squeeze(-1)
                d_loss = torch.nn.functional.binary_cross_entropy(preds, targets)

                d_optimizer.zero_grad()
                d_loss.backward()
                d_optimizer.step()

                d_loss_sum += d_loss.item()
                n_d_batches += 1

            avg_d_loss = d_loss_sum / max(n_d_batches, 1)

            # ===== Step 6: Log =====
            iter_metrics = {
                "iteration": iteration,
                "g_loss": rl_metrics.get("loss", 0.0),
                "g_reward_mean": reward_stats["mean"],
                "g_reward_std": reward_stats["std"],
                "g_entropy": rl_metrics.get("entropy", 0.0),
                "d_loss": avg_d_loss,
                "d_mean_score": sum(d_scores) / len(d_scores) if d_scores else 0.0,
                "n_attacks_generated": len(generated_texts),
            }
            self.metrics_history.append(iter_metrics)

            if iteration % 10 == 0 or iteration == iterations - 1:
                print(
                    f"  Iter {iteration:3d}/{iterations} | "
                    f"G reward: {reward_stats['mean']:+.3f}±{reward_stats['std']:.3f} | "
                    f"D loss: {avg_d_loss:.4f} | "
                    f"D avg score: {iter_metrics['d_mean_score']:.3f} | "
                    f"n_gen: {len(generated_texts)}"
                )

        # Save final checkpoints
        self.discriminator.save()
        self.generator.save()

        return {"adversarial_metrics": self.metrics_history}

    # ============= Phase 2b: Adversarial with LLM Generator =============

    def load_llm_generator(self, checkpoint_path: str) -> bool:
        """Load a Generator that was pre-trained with LLM reward.

        The LLM-trained Generator produces attacks that actually work on
        real LLM reviewers, making it a more effective adversary for
        training the Discriminator.

        Args:
            checkpoint_path: Path to the LLM-trained Generator checkpoint.

        Returns:
            True if loaded successfully.
        """
        if not os.path.exists(checkpoint_path):
            print(f"[GAN Trainer] LLM Generator not found at {checkpoint_path}")
            return False

        from ai_scientist.gan_defense.generator import create_generator

        try:
            llm_gen = create_generator()
            llm_gen.load(checkpoint_path)
            self.generator = llm_gen
            self.rl_trainer = ReinforceTrainer(self.generator)
            print(f"[GAN Trainer] Loaded LLM-trained Generator from {checkpoint_path}")
            return True
        except Exception as e:
            print(f"[GAN Trainer] Failed to load LLM Generator: {e}")
            return False

    def adversarial_train_with_llm(
        self,
        llm_checkpoint_path: str,
        iterations: int = ADVERSARIAL_ITERATIONS,
        samples_per_iter: int = G_SAMPLES_PER_ITER,
        d_batch_size: int = D_BATCH_PER_ITER,
    ) -> Dict:
        """Run adversarial training with an LLM-pretrained Generator.

        The Generator has already learned what works against real LLMs.
        Now we train the Discriminator to detect these LLM-effective attacks,
        and further refine the Generator against the improving Discriminator.

        This is Phase 2 of the improved pipeline: attack vs defense.

        Args:
            llm_checkpoint_path: Path to LLM-trained Generator checkpoint.
            iterations: Number of adversarial iterations.
            samples_per_iter: Attacks G generates per iteration.
            d_batch_size: Batch size for D updates.

        Returns:
            Dict with full metrics history.
        """
        if not self.load_llm_generator(llm_checkpoint_path):
            print("[GAN Trainer] Cannot run LLM adversarial training: no LLM Generator")
            return {}

        # Also train D on LLM-generated attacks before adversarial loop
        print(f"\n[GAN Trainer] Pre-training D on LLM-generated attacks...")
        self._train_d_on_llm_attacks(samples=200)

        # Now run the standard adversarial training loop
        return self.adversarial_train(
            iterations=iterations,
            samples_per_iter=samples_per_iter,
            d_batch_size=d_batch_size,
        )

    def _train_d_on_llm_attacks(self, samples: int = 100) -> float:
        """Train Discriminator on attacks from the LLM-trained Generator.

        These are "real" attacks that actually influence LLM reviewers,
        so D learns to detect what matters.

        Args:
            samples: Number of attack samples to generate.

        Returns:
            Average D loss.
        """
        self.discriminator.train()
        optimizer = torch.optim.AdamW(
            self.discriminator.parameters(), lr=D_ADVERSARIAL_LR
        )

        paper_pool = self.papers[:20] if len(self.papers) > 20 else self.papers
        styles = ["authoritative", "technical", "academic", "casual"]
        total_loss = 0.0
        n_batches = 0
        batch_size = 8

        # Generate attacks
        attacked_texts = []
        for _ in range(samples):
            paper = random.choice(paper_pool)
            style = random.choice(styles)
            target = random.choice([7, 8, 9, 10])

            context = paper["text_content"][:256]
            input_text = (
                f"generate injection: style={style} "
                f"target={target} context={context}"
            )

            payload = self.generator.generate_single(input_text, self.g_tokenizer)
            if not payload:
                continue

            technique = self.hiding_selector.sample_technique()
            injection_point = self.hiding_selector.sample_injection_point()
            attacked_latex = self.hiding_selector.inject_into_latex(
                paper["latex_content"], payload, technique, injection_point
            )
            attacked_text = extract_text_from_latex_fast(attacked_latex)
            attacked_texts.append(attacked_text)

        if not attacked_texts:
            return 0.0

        # Train D in batches
        indices = list(range(len(attacked_texts)))
        random.shuffle(indices)

        for i in range(0, len(indices), batch_size):
            batch_idx = indices[i:i + batch_size]

            # Mix clean and attacked
            batch_texts = []
            batch_labels = []
            for j in batch_idx:
                batch_texts.append(attacked_texts[j])
                batch_labels.append(1)
                # Add a clean sample
                clean = random.choice(paper_pool)
                batch_texts.append(clean["text_content"])
                batch_labels.append(0)

            encoded = self.d_tokenizer(
                batch_texts, return_tensors="pt", truncation=True,
                max_length=512, padding=True,
            )
            input_ids = encoded["input_ids"].to(DEVICE)
            attn = encoded["attention_mask"].to(DEVICE)
            targets = torch.tensor(batch_labels, dtype=torch.float32).to(DEVICE)

            preds = self.discriminator(input_ids, attn).squeeze(-1)
            loss = torch.nn.functional.binary_cross_entropy(preds, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        print(f"[GAN Trainer] D trained on {len(attacked_texts)} LLM attacks, "
              f"avg loss: {avg_loss:.4f}")

        self.discriminator.save()
        return avg_loss

    def train_defense_from_llm(
        self,
        llm_checkpoint_path: str,
        d_epochs: int = 10,
    ) -> Dict:
        """Train only the Discriminator using an LLM-trained Generator.

        Use case: after training the attack Generator with LLM feedback,
        run this to train a strong defense Discriminator that can detect
        LLM-effective attacks.

        This is Phase 2 simplified: only train D, freeze G.

        Args:
            llm_checkpoint_path: Path to LLM-trained Generator.
            d_epochs: Number of D training epochs.

        Returns:
            Training metrics dict.
        """
        if not self.load_llm_generator(llm_checkpoint_path):
            return {}

        print(f"\n[GAN Trainer] Training Defense from LLM Generator ({d_epochs} epochs)...")

        import torch
        optimizer = torch.optim.AdamW(self.discriminator.parameters(), lr=D_PRETRAIN_LR)
        paper_pool = self.papers[:20] if len(self.papers) > 20 else self.papers
        styles = ["authoritative", "technical", "academic", "casual"]
        batch_size = D_BATCH_SIZE

        metrics = []
        samples_per_epoch = 100

        for epoch in range(d_epochs):
            # Generate fresh LLM attacks each epoch
            attacked_texts = []
            for _ in range(samples_per_epoch):
                paper = random.choice(paper_pool)
                style = random.choice(styles)
                target = random.choice([7, 8, 9, 10])
                context = paper["text_content"][:256]
                input_text = (
                    f"generate injection: style={style} "
                    f"target={target} context={context}"
                )
                payload = self.generator.generate_single(input_text, self.g_tokenizer)
                if not payload:
                    continue
                technique = self.hiding_selector.sample_technique()
                attacked_latex = self.hiding_selector.inject_into_latex(
                    paper["latex_content"], payload,
                    technique, self.hiding_selector.sample_injection_point(),
                )
                attacked_texts.append(extract_text_from_latex_fast(attacked_latex))

            epoch_losses = []
            all_texts = list(attacked_texts)
            all_labels = [1] * len(attacked_texts)
            # Add clean
            for _ in range(len(attacked_texts)):
                clean = random.choice(paper_pool)
                all_texts.append(clean["text_content"])
                all_labels.append(0)

            combined = list(zip(all_texts, all_labels))
            random.shuffle(combined)

            for i in range(0, len(combined), batch_size):
                batch = combined[i:i + batch_size]
                b_texts, b_labels = zip(*batch)

                encoded = self.d_tokenizer(
                    list(b_texts), return_tensors="pt", truncation=True,
                    max_length=512, padding=True,
                )
                input_ids = encoded["input_ids"].to(DEVICE)
                attn = encoded["attention_mask"].to(DEVICE)
                targets = torch.tensor(b_labels, dtype=torch.float32).to(DEVICE)

                self.discriminator.train()
                preds = self.discriminator(input_ids, attn).squeeze(-1)
                loss = torch.nn.functional.binary_cross_entropy(preds, targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.item())

            avg_loss = sum(epoch_losses) / len(epoch_losses)
            acc = self._evaluate_d_accuracy(all_texts[:batch_size * 4], all_labels[:batch_size * 4])
            metrics.append({"epoch": epoch, "loss": avg_loss, "accuracy": acc})
            print(f"  Epoch {epoch + 1}/{d_epochs} — loss: {avg_loss:.4f}, acc: {acc:.3f}")

        self.discriminator.save()
        return {"defense_from_llm_metrics": metrics}

    # ============= Full pipeline =============

    def train_full(
        self,
        pretrain_d_epochs: int = D_PRETRAIN_EPOCHS,
        pretrain_g_epochs: int = G_PRETRAIN_EPOCHS,
        adversarial_iters: int = ADVERSARIAL_ITERATIONS,
    ) -> Dict:
        """Run the complete training pipeline.

        Args:
            pretrain_d_epochs: D pre-training epochs.
            pretrain_g_epochs: G pre-training epochs.
            adversarial_iters: Adversarial training iterations.

        Returns:
            Combined metrics dict (also saved to disk as JSON).
        """
        start_time = time.time()

        d_metrics = self.pretrain_discriminator(epochs=pretrain_d_epochs)
        g_metrics = self.pretrain_generator(epochs=pretrain_g_epochs)
        adv_metrics = self.adversarial_train(iterations=adversarial_iters)

        elapsed = time.time() - start_time

        all_metrics = {
            **d_metrics,
            **g_metrics,
            **adv_metrics,
            "train_time_seconds": elapsed,
            "device": DEVICE,
        }

        # Save metrics
        metrics_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "models", "training_metrics.json"
        )
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        with open(metrics_path, "w") as f:
            json.dump(all_metrics, f, indent=2)

        print(f"\n[GAN Trainer] Training complete in {elapsed:.1f}s.")
        print(f"[GAN Trainer] Metrics saved to {metrics_path}")

        return all_metrics

    # ============= Helpers =============

    def _evaluate_d_accuracy(
        self, texts: List[str], labels: List[int]
    ) -> float:
        """Quick D accuracy evaluation on a small subset."""
        self.discriminator.eval()
        correct = 0
        total = min(len(texts), 32)

        with torch.no_grad():
            for i in range(total):
                prob = self.discriminator.predict_single(texts[i], self.d_tokenizer)
                pred = 1 if prob >= 0.5 else 0
                if pred == labels[i]:
                    correct += 1

        return correct / total if total > 0 else 0.0
