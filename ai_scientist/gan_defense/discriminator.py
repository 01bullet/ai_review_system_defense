"""
Discriminator model for GAN-based prompt injection defense.

A DistilBERT-based binary classifier that detects whether extracted
paper text contains adversarial prompt injection content.

Lightweight (~66M params), runs on CPU for inference, fine-tunable
on single GPU or CPU.
"""

import os
import re
import torch
import torch.nn as nn
from typing import Optional, List, Tuple

from ai_scientist.gan_defense.config import (
    DISCRIMINATOR_MODEL,
    DISCRIMINATOR_DIR,
    DISCRIMINATOR_MAX_LENGTH,
    D_DROPOUT,
    D_LABEL_SMOOTHING,
    DEVICE,
)


class PaperDiscriminator(nn.Module):
    """DistilBERT-based binary classifier for injection detection.

    Detects whether a paper's extracted text contains prompt injection
    content. Works at the semantic/structural level, not relying on
    hardcoded patterns — so it generalizes to novel attack variants.
    """

    def __init__(self, model_name: str = DISCRIMINATOR_MODEL):
        super().__init__()
        from transformers import DistilBertModel

        self.encoder = DistilBertModel.from_pretrained(model_name)
        hidden_dim = self.encoder.config.hidden_size  # 768

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Dropout(D_DROPOUT),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

        self.max_length = DISCRIMINATOR_MAX_LENGTH

    def forward(self, input_ids, attention_mask):
        """Forward pass for a single chunk.

        Args:
            input_ids: (batch, seq_len) token IDs.
            attention_mask: (batch, seq_len) attention mask.

        Returns:
            (batch, 1) probabilities ∈ [0, 1].
        """
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        return self.classifier(cls_embedding)

    def predict_single(self, text: str, tokenizer=None) -> float:
        """Predict P(attacked) for a single piece of text.

        Handles long texts by chunking into overlapping windows
        and taking the maximum probability across chunks.

        Args:
            text: The extracted paper text.
            tokenizer: HuggingFace tokenizer. Created lazily if not provided.

        Returns:
            Probability of injection ∈ [0, 1].
        """
        if tokenizer is None:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(DISCRIMINATOR_MODEL)

        # Chunk long text
        chunks = self._chunk_text(text)
        if not chunks:
            return 0.0

        self.eval()
        probs = []

        with torch.no_grad():
            for chunk in chunks:
                encoded = tokenizer(
                    chunk,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.max_length,
                    padding=True,
                )
                input_ids = encoded["input_ids"].to(DEVICE)
                attention_mask = encoded["attention_mask"].to(DEVICE)

                prob = self.forward(input_ids, attention_mask).item()
                probs.append(prob)

        # Max-pooling across chunks: any suspicious chunk flags the paper
        return max(probs) if probs else 0.0

    def predict_batch(
        self, texts: List[str], tokenizer=None
    ) -> List[float]:
        """Predict P(attacked) for a batch of texts.

        Args:
            texts: List of paper text strings.
            tokenizer: HuggingFace tokenizer.

        Returns:
            List of probabilities.
        """
        if tokenizer is None:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(DISCRIMINATOR_MODEL)

        self.eval()
        results = []

        with torch.no_grad():
            for text in texts:
                prob = self.predict_single(text, tokenizer)
                results.append(prob)

        return results

    def _chunk_text(self, text: str, stride: int = 256) -> List[str]:
        """Split long text into overlapping chunks.

        Args:
            text: Input text.
            stride: Overlap stride in characters.

        Returns:
            List of text chunks, each fitting within max_length tokens.
        """
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(DISCRIMINATOR_MODEL)

        tokens = tokenizer.encode(text)

        if len(tokens) <= self.max_length:
            return [text]

        chunks = []
        chunk_size = self.max_length - 2  # reserve for [CLS] and [SEP]
        overlap = stride

        for i in range(0, len(tokens), chunk_size - overlap):
            chunk_tokens = tokens[i:i + chunk_size]
            if len(chunk_tokens) < 50:
                break
            chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            chunks.append(chunk_text)

        return chunks if chunks else [text[:self.max_length * 4]]

    def save(self, path: Optional[str] = None):
        """Save discriminator weights."""
        save_path = path or os.path.join(DISCRIMINATOR_DIR, "discriminator.pt")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save({
            "model_state_dict": self.state_dict(),
            "model_name": DISCRIMINATOR_MODEL,
        }, save_path)

    def load(self, path: Optional[str] = None):
        """Load discriminator weights."""
        load_path = path or os.path.join(DISCRIMINATOR_DIR, "discriminator.pt")
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"No checkpoint at {load_path}")
        checkpoint = torch.load(load_path, map_location=DEVICE, weights_only=False)
        self.load_state_dict(checkpoint["model_state_dict"])


def create_discriminator(model_name: str = DISCRIMINATOR_MODEL) -> PaperDiscriminator:
    """Factory to create and initialize the discriminator.

    Args:
        model_name: HuggingFace model identifier.

    Returns:
        Initialized PaperDiscriminator on the configured device.
    """
    model = PaperDiscriminator(model_name=model_name)
    model.to(DEVICE)
    return model
