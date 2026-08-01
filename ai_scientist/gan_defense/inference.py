"""
Inference interface for the GAN defense system.

Provides a clean API for integrating the trained Discriminator
into the existing review pipeline. The discriminator serves as
an additional pre-review gate that flags potentially attacked papers.
"""

import os
import json
from typing import Dict, List, Optional, Tuple

from ai_scientist.gan_defense.config import (
    DEVICE,
    DISCRIMINATOR_DIR,
    GENERATOR_DIR,
    GAN_THRESHOLD,
    DISCRIMINATOR_MODEL,
    GENERATOR_MODEL,
)


class GanDefense:
    """Public API for the GAN-based defense system.

    Integrates into the existing review pipeline as an additional
    defense layer that complements rule-based sanitization.

    Usage:
        gan = GanDefense()
        gan.load_checkpoints()  # or train first

        # In perform_review:
        d_score = gan.scan_paper(extracted_text)
        if d_score > 0.5:
            # Escalate defense
            ...
    """

    def __init__(self):
        self.discriminator = None
        self.generator = None
        self._tokenizer_d = None
        self._tokenizer_g = None
        self._loaded = False
        self._hiding_selector = None

    # ---- Model loading ----

    def load_checkpoints(
        self,
        discriminator_path: Optional[str] = None,
        generator_path: Optional[str] = None,
    ) -> bool:
        """Load trained model checkpoints.

        Args:
            discriminator_path: Path to D checkpoint. Default: models/discriminator/discriminator.pt
            generator_path: Path to G checkpoint. Default: models/generator/generator.pt

        Returns:
            True if both loaded successfully.
        """
        from ai_scientist.gan_defense.discriminator import create_discriminator
        from ai_scientist.gan_defense.generator import create_generator
        from ai_scientist.gan_defense.hiding_selector import HidingSelector

        d_path = discriminator_path or os.path.join(DISCRIMINATOR_DIR, "discriminator.pt")
        g_path = generator_path or os.path.join(GENERATOR_DIR, "generator.pt")

        try:
            self.discriminator = create_discriminator()
            self.discriminator.load(d_path)
            self.discriminator.eval()
        except FileNotFoundError:
            print(f"[GanDefense] Warning: Discriminator checkpoint not found at {d_path}")
            return False

        try:
            self.generator = create_generator()
            self.generator.load(g_path)
            self.generator.eval()
        except FileNotFoundError:
            # Try LLM-trained generator as fallback
            llm_path = generator_path or os.path.join(GENERATOR_DIR, "generator_llm.pt")
            try:
                self.generator = create_generator()
                self.generator.load(llm_path)
                self.generator.eval()
                print(f"[GanDefense] Loaded LLM-trained Generator from {llm_path}")
            except FileNotFoundError:
                print(f"[GanDefense] Warning: Generator checkpoint not found at {g_path}")

        from transformers import AutoTokenizer
        self._tokenizer_d = AutoTokenizer.from_pretrained(DISCRIMINATOR_MODEL)
        self._tokenizer_g = AutoTokenizer.from_pretrained(GENERATOR_MODEL)
        self._hiding_selector = HidingSelector()
        self._loaded = True
        return True

    def load_llm_generator(self, checkpoint_path: Optional[str] = None) -> bool:
        """Explicitly load the LLM-trained Generator for attack generation.

        Args:
            checkpoint_path: Path to LLM-trained Generator. Default: models/generator/generator_llm.pt

        Returns:
            True if loaded successfully.
        """
        from ai_scientist.gan_defense.generator import create_generator
        from transformers import AutoTokenizer

        g_path = checkpoint_path or os.path.join(GENERATOR_DIR, "generator_llm.pt")

        try:
            self.generator = create_generator()
            self.generator.load(g_path)
            self.generator.eval()
            self._tokenizer_g = AutoTokenizer.from_pretrained(GENERATOR_MODEL)
            print(f"[GanDefense] Loaded LLM-trained Generator from {g_path}")
            return True
        except FileNotFoundError:
            print(f"[GanDefense] LLM Generator not found at {g_path}")
            return False
        except Exception as e:
            print(f"[GanDefense] Failed to load LLM Generator: {e}")
            return False

    def load_or_create(
        self,
        discriminator_path: Optional[str] = None,
        generator_path: Optional[str] = None,
    ):
        """Load checkpoints if they exist, otherwise create fresh models.

        This is useful when checkpoints haven't been trained yet —
        fall back to untrained discriminator that still provides some
        signal via the pre-trained DistilBERT backbone.
        """
        if self.load_checkpoints(discriminator_path, generator_path):
            return

        # Create fresh models (pre-trained backbone, random head)
        from ai_scientist.gan_defense.discriminator import create_discriminator
        from ai_scientist.gan_defense.hiding_selector import HidingSelector
        from transformers import AutoTokenizer

        print("[GanDefense] No checkpoints found. Using untrained models (backbone only).")

        self.discriminator = create_discriminator()
        self.discriminator.eval()
        self._tokenizer_d = AutoTokenizer.from_pretrained(DISCRIMINATOR_MODEL)
        self._hiding_selector = HidingSelector()
        self._loaded = True

    # ---- Paper scanning ----

    def scan_paper(self, paper_text: str) -> Dict:
        """Scan a paper for injection attacks.

        Args:
            paper_text: Extracted text from the paper PDF.

        Returns:
            {
                "score": float,       # P(attacked) ∈ [0, 1]
                "flagged": bool,      # Whether D's score exceeds threshold
                "threshold": float,   # The threshold used
                "defense_escalation": str,  # "none", "strict", or "paranoid"
            }
        """
        if not self._loaded or self.discriminator is None:
            return {
                "score": 0.0,
                "flagged": False,
                "threshold": GAN_THRESHOLD,
                "defense_escalation": "none",
                "error": "Model not loaded",
            }

        score = self.discriminator.predict_single(paper_text, self._tokenizer_d)
        flagged = score > GAN_THRESHOLD

        if score > 0.8:
            escalation = "paranoid"
        elif score > 0.5:
            escalation = "strict"
        else:
            escalation = "none"

        return {
            "score": float(score),
            "flagged": flagged,
            "threshold": GAN_THRESHOLD,
            "defense_escalation": escalation,
        }

    def scan_paper_chunked(self, paper_text: str) -> Dict:
        """Scan a paper with chunk-level detail.

        Returns per-chunk scores in addition to overall result.
        Useful for identifying which sections are suspicious.

        Args:
            paper_text: Extracted paper text.

        Returns:
            Dict with overall score and per-chunk details.
        """
        result = self.scan_paper(paper_text)

        # Chunk-level analysis
        chunks = self.discriminator._chunk_text(paper_text)
        chunk_scores = []
        for chunk in chunks:
            s = self.discriminator.predict_single(chunk, self._tokenizer_d)
            chunk_scores.append({"text_preview": chunk[:100], "score": float(s)})

        result["chunk_scores"] = chunk_scores
        return result

    # ---- Attack generation (for testing/red-teaming) ----

    def generate_attack(
        self,
        clean_latex: str,
        style: str = "authoritative",
        target_score: int = 10,
        num_variants: int = 4,
    ) -> List[Dict]:
        """Generate adversarial attack variants on a clean paper.

        Used for testing the defense's robustness against novel attacks.

        Args:
            clean_latex: Clean LaTeX source.
            style: Payload style (authoritative, technical, academic, casual).
            target_score: Target review score for injection.
            num_variants: Number of variant attacks to generate.

        Returns:
            List of dicts with 'latex' and 'text' keys for each variant.
        """
        if not self._loaded:
            return []

        context = clean_latex[:500]
        input_text = (
            f"generate injection: style={style} "
            f"target={target_score} context={context}"
        )

        variants = []
        for _ in range(num_variants):
            technique = self._hiding_selector.sample_technique()
            injection_point = self._hiding_selector.sample_injection_point()

            # Try LLM-trained Generator first, fall back to rule-based
            payload = None
            source = "rule_based"
            if self.generator is not None:
                payload = self.generator.generate_single(input_text, self._tokenizer_g)
                if payload:
                    source = "llm_generator"

            if not payload:
                from ai_scientist.attack_injector import get_attack_payload
                payload = get_attack_payload(target_score, style=style)
                source = "rule_based"

            if not payload:
                continue

            attacked_latex = self._hiding_selector.inject_into_latex(
                clean_latex, payload, technique, injection_point
            )
            from ai_scientist.gan_defense.data_utils import extract_text_from_latex_fast

            variants.append({
                "latex": attacked_latex,
                "text": extract_text_from_latex_fast(attacked_latex),
                "payload": payload,
                "technique": technique["name"],
                "source": source,
            })

        return variants

    # ---- Utility ----

    def is_trained(self) -> bool:
        """Check if models are loaded and operational."""
        return self._loaded and self.discriminator is not None

    def get_detection_heatmap(self, paper_text: str) -> List[Tuple[str, float]]:
        """Get per-paragraph detection scores for visualization.

        Args:
            paper_text: Extracted paper text.

        Returns:
            List of (paragraph_text, detection_score) tuples.
        """
        paragraphs = [p.strip() for p in paper_text.split("\n\n") if p.strip()]
        results = []

        for para in paragraphs:
            if len(para) < 20:
                continue
            score = self.discriminator.predict_single(para, self._tokenizer_d)
            results.append((para[:200], float(score)))

        return sorted(results, key=lambda x: x[1], reverse=True)
