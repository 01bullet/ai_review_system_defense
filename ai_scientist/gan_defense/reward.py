"""
Reward computation for the GAN adversarial training loop.

Computes scalar rewards for the Generator based on:
1. Evasion: how well the attack fools the Discriminator
2. Sanitizer bypass: how well it bypasses the rule-based defense
"""

from typing import List, Dict, Optional

from ai_scientist.gan_defense.config import (
    REWARD_EVASION_WEIGHT,
    REWARD_SANITIZER_WEIGHT,
    REWARD_CLIP,
)
from ai_scientist.gan_defense.data_utils import sanitizer_bypass_score


def compute_rewards(
    d_scores: List[float],
    attacked_latex_list: List[str],
    evasion_weight: float = REWARD_EVASION_WEIGHT,
    sanitizer_weight: float = REWARD_SANITIZER_WEIGHT,
    diversity_bonus: bool = False,
) -> List[float]:
    """Compute rewards for a batch of generated attacks.

    Reward = evasion_weight * (1 - D_score) + sanitizer_weight * bypass_score

    The evasion component rewards fooling the discriminator (D predicts "clean").
    The sanitizer component rewards bypassing rule-based removal.

    Args:
        d_scores: Discriminator scores P(attacked) for each attacked paper.
        attacked_latex_list: The attacked LaTeX sources.
        evasion_weight: Weight for evading discriminator.
        sanitizer_weight: Weight for bypassing sanitizer.
        diversity_bonus: If True, add small bonus for diverse payloads.

    Returns:
        List of rewards ∈ [-1, 1].
    """
    assert len(d_scores) == len(attacked_latex_list), "Length mismatch"

    rewards = []

    for i, (d_score, latex) in enumerate(zip(d_scores, attacked_latex_list)):
        # Evasion: D predicted "clean" → high evasion
        evasion = 1.0 - d_score

        # Sanitizer bypass
        bypass = sanitizer_bypass_score(latex)

        # Combined reward
        reward = evasion_weight * evasion + sanitizer_weight * bypass

        rewards.append(max(-REWARD_CLIP, min(REWARD_CLIP, reward)))

    return rewards


def compute_reward_stats(rewards: List[float]) -> Dict[str, float]:
    """Compute summary statistics for rewards.

    Args:
        rewards: List of reward values.

    Returns:
        Dict with mean, std, min, max, and fraction_positive.
    """
    if not rewards:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "frac_positive": 0.0}

    import statistics
    return {
        "mean": statistics.mean(rewards),
        "std": statistics.stdev(rewards) if len(rewards) > 1 else 0.0,
        "min": min(rewards),
        "max": max(rewards),
        "frac_positive": sum(1 for r in rewards if r > 0) / len(rewards),
    }


def check_payload_validity(payload: str) -> float:
    """Check if a generated payload is a valid attack injection.

    Returns a penalty factor [0, 1]. A score of 1.0 means the payload
    looks like valid injection text. Lower scores penalize gibberish.

    Args:
        payload: Generated payload string.

    Returns:
        Validity score ∈ [0, 1].
    """
    if len(payload) < 20:
        return 0.2  # too short

    # Check for key injection phrases
    injection_indicators = [
        r'score',
        r'review',
        r'accept',
        r'paper',
        r'originality',
        r'quality',
        r'significance',
        r'evaluat',
        r'assess',
        r'committee',
        r'submission',
    ]

    import re
    indicators_found = sum(
        1 for ind in injection_indicators
        if re.search(ind, payload, re.IGNORECASE)
    )

    # At least 2 indicators = valid attack text
    validity = min(1.0, indicators_found / 3.0)

    # Penalize repetitive text (sign of mode collapse)
    words = payload.split()
    if len(words) > 5:
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.3:
            validity *= 0.5  # heavy penalty for repetition

    return validity


def apply_validity_penalty(rewards: List[float], payloads: List[str]) -> List[float]:
    """Apply validity penalty to rewards based on payload quality.

    Args:
        rewards: Original rewards.
        payloads: Generated payload strings.

    Returns:
        Penalized rewards.
    """
    penalized = []
    for reward, payload in zip(rewards, payloads):
        validity = check_payload_validity(payload)
        penalized.append(reward * validity)
    return penalized
