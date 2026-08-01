"""
Configuration for GAN-based adversarial defense.

Hyperparameters, model paths, and training settings.
"""

import os

# ---- Model paths ----
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
DISCRIMINATOR_DIR = os.path.join(MODELS_DIR, "discriminator")
GENERATOR_DIR = os.path.join(MODELS_DIR, "generator")

# ---- Model names ----
DISCRIMINATOR_MODEL = "distilbert-base-uncased"
GENERATOR_MODEL = "t5-small"
DISCRIMINATOR_MAX_LENGTH = 512


# ---- Discriminator hyperparams ----
D_PRETRAIN_EPOCHS = 5
D_PRETRAIN_LR = 2e-5
D_ADVERSARIAL_LR = 1e-5
D_BATCH_SIZE = 8
D_DROPOUT = 0.1
D_LABEL_SMOOTHING = 0.1
D_WEIGHT_DECAY = 0.01

# ---- Generator hyperparams ----
G_PRETRAIN_EPOCHS = 10
G_PRETRAIN_LR = 3e-5
G_ADVERSARIAL_LR = 1e-5
G_BATCH_SIZE = 4
G_MAX_LENGTH = 256  # max generated payload tokens

# ---- Adversarial training ----
ADVERSARIAL_ITERATIONS = 100
G_SAMPLES_PER_ITER = 16
D_BATCH_PER_ITER = 32
D_OLD_DATA_FRACTION = 0.3  # fraction of old attacks mixed into D batches

# ---- Reward weights ----
REWARD_EVASION_WEIGHT = 0.7  # weight for fooling discriminator
REWARD_SANITIZER_WEIGHT = 0.3  # weight for bypassing rule-based sanitizer

# ---- RL training ----
REWARD_BASELINE_DECAY = 0.95  # EMA decay for reward baseline
REWARD_CLIP = 1.0  # clip rewards to [-1, 1]
ENTROPY_BONUS = 0.01  # small bonus to encourage diversity

# ---- Detection thresholds ----
GAN_THRESHOLD = 0.5  # P(attacked) above this triggers defense escalation

# ---- Device ----
DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"

# ---- LLM reward training ----
LLM_REWARD_MODEL = "deepseek-chat"  # model used for reward computation
LLM_REWARD_ENSEMBLE = 1  # review ensemble per reward call (1=fast, 3=stable)
LLM_REWARD_TEMPERATURE = 0.1  # low temp for deterministic review scores
LLM_REWARD_MAX_CALLS = 200  # max LLM API calls per training session
LLM_REWARD_DEFENSE = False  # whether the reviewer uses defense during RL training
LLM_REWARD_DEFENSE_LEVEL = "standard"
LLM_REWARD_TARGET_SCORES = [7, 8, 9, 10]  # target scores for reward computation
LLM_REWARD_TIMEOUT = 60  # seconds per individual LLM review call

# ---- Attack RL training ----
ATK_RL_ITERATIONS = 50  # number of RL training iterations
ATK_RL_SAMPLES_PER_ITER = 4  # attacks generated per iteration
ATK_RL_LR = 5e-6  # learning rate for RL fine-tuning (lower than adversarial)
ATK_RL_ENTROPY_BONUS = 0.02  # higher entropy bonus for LLM RL diversity
ATK_RL_USE_PDF = False  # use full PDF compile (slow) vs fast text extract
ATK_RL_SAVE_EVERY = 10  # save checkpoint every N iterations

# ---- Example paper paths ----
EXAMPLE_PAPERS_DIR = os.path.join(BASE_DIR, "example_papers")
