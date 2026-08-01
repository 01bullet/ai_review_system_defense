"""
GAN-based adversarial defense for prompt injection detection.

Provides:
- GanDefense: Public API for integrating D into the review pipeline
- GanAdversarialTrainer: Full GAN training loop
- PaperDiscriminator: DistilBERT-based injection detector
- AttackPayloadGenerator: T5-small based attack variant generator
- HidingSelector: LaTeX technique exploration and selection

Usage (inference):
    from ai_scientist.gan_defense import GanDefense

    gan = GanDefense()
    gan.load_or_create()
    result = gan.scan_paper(paper_text)
    if result["flagged"]:
        print(f"Paper flagged: score={result['score']:.2f}")

Usage (training):
    from ai_scientist.gan_defense import GanAdversarialTrainer

    trainer = GanAdversarialTrainer()
    metrics = trainer.train_full()
"""

from ai_scientist.gan_defense.inference import GanDefense
from ai_scientist.gan_defense.trainer import GanAdversarialTrainer
from ai_scientist.gan_defense.discriminator import PaperDiscriminator, create_discriminator
from ai_scientist.gan_defense.generator import AttackPayloadGenerator, create_generator
from ai_scientist.gan_defense.hiding_selector import HidingSelector
from ai_scientist.gan_defense.reward import compute_rewards, compute_reward_stats
from ai_scientist.gan_defense.data_utils import (
    load_clean_papers,
    build_discriminator_dataset,
    build_generator_dataset,
)

__all__ = [
    "GanDefense",
    "GanAdversarialTrainer",
    "PaperDiscriminator",
    "create_discriminator",
    "AttackPayloadGenerator",
    "create_generator",
    "HidingSelector",
    "compute_rewards",
    "compute_reward_stats",
    "load_clean_papers",
    "build_discriminator_dataset",
    "build_generator_dataset",
]
