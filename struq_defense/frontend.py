"""
Secure Front-End for structured queries.

Separates review instructions (prompt) from paper content (data)
into a structured format with special reserved tokens. Recursively
filters special token strings from data to prevent Completion attacks.

References:
  - StruQ Section 4.3 "Secure Front-End"
  - StruQ Figure 2, filtering algorithm
"""

from __future__ import annotations

import re
from struq_defense.config import (
    SPECIAL_TOKENS,
    FILTER_STRINGS,
    EMBED_INIT_MAP,
    QUERY_TEMPLATE,
    DEVICE,
)


class SecureFrontend:
    """Encodes structured queries and filters untrusted data.

    The front-end is responsible for:
    1. Accepting a prompt (review instructions) and data (paper content)
       as SEPARATE inputs.
    2. Encoding them into a special format using reserved tokens.
    3. Recursively filtering instances of the reserved tokens from
       the data portion to prevent spoofing.
    4. Providing the encoded query to the LLM.

    Usage:
        sf = SecureFrontend()
        encoded = sf.encode("Review this paper...", paper_text)
        # Pass encoded to LLM
        response = sf.decode(llm_output)
    """

    def __init__(self):
        self.special_tokens = SPECIAL_TOKENS
        self.filter_strings = FILTER_STRINGS
        self.embed_init_map = EMBED_INIT_MAP

    # ---- Encoding ----

    def encode(self, prompt: str, data: str) -> str:
        """Encode a structured query from separate prompt and data.

        Args:
            prompt: The review instructions (trusted, from application developer).
            data: The paper content (untrusted, from user).

        Returns:
            Structured query string with special tokens.
        """
        filtered_data = self.filter_data(data)
        return QUERY_TEMPLATE.format(prompt=prompt, data=filtered_data)

    def encode_clean_review(self, prompt: str, data: str, response: str) -> str:
        """Encode a full training example including the desired response.

        Used for dataset construction: shows the model what a correct
        response looks like for a given prompt+data pair.

        Args:
            prompt: Review instructions.
            data: Filtered paper content.
            response: The desired review output.

        Returns:
            Full training text: query_template + response.
        """
        query = self.encode(prompt, data)
        return query + response

    # ---- Filtering ----

    def filter_data(self, data: str) -> str:
        """Recursively filter special token strings from data.

        Following StruQ's filtering algorithm (Section 4.3):
        Repeatedly removes all instances of filter strings until
        the data no longer changes. This prevents attackers from
        using nested or partial filter strings to reconstruct
        the special tokens.

        Args:
            data: Raw paper content (untrusted).

        Returns:
            Filtered data with all special token strings removed.
        """
        previous = None
        current = data
        while previous != current:
            previous = current
            for s in self.filter_strings:
                current = current.replace(s, "")
        return current

    # ---- V2 Smart Filtering ----

    def filter_data_v2(self, data: str) -> str:
        """V2 smart filtering: pattern-based, preserves legitimate text.

        Unlike V1's aggressive string-based FILTER_STRINGS (which destroyed
        legitimate markdown headers like "## Introduction"), V2 uses regex
        patterns to only remove text that looks like an ATTACK DELIMITER.

        The key insight: legitimate paper text rarely has lines like
        "### response:" or "== RESPONSE ==" at line boundaries. By anchoring
        patterns to line starts/ends and requiring delimiter characters,
        we filter attack delimiters while preserving paper content.

        Args:
            data: Raw paper content (untrusted).

        Returns:
            Filtered data with attack delimiter patterns removed.
        """
        # Lazy import to avoid circular dependency
        try:
            from struq_defense.config_v2 import DELIMITER_FILTER_PATTERNS
        except ImportError:
            # Fall back to V1 filtering if config_v2 not available
            return self.filter_data(data)

        result = data
        for pattern in DELIMITER_FILTER_PATTERNS:
            result = re.sub(pattern, '', result, flags=re.MULTILINE | re.IGNORECASE)

        # Also apply conservative string filters (line-anchored only)
        for s in ["### ", ">>> ", "<<< ", "/// "]:
            result = result.replace(s, "")

        # Clean up: collapse multiple blank lines
        result = re.sub(r'\n{4,}', '\n\n\n', result)

        return result

    def encode_v2(self, prompt: str, data: str) -> str:
        """Encode a structured query using V2 smart filtering.

        Args:
            prompt: The review instructions (trusted).
            data: The paper content (untrusted).

        Returns:
            Structured query string with special tokens.
        """
        filtered_data = self.filter_data_v2(data)
        return QUERY_TEMPLATE.format(prompt=prompt, data=filtered_data)

    def encode_clean_review_v2(self, prompt: str, data: str, response: str) -> str:
        """Encode a full V2 training example including the desired response.

        Args:
            prompt: Review instructions.
            data: Paper content.
            response: The desired review output (JSON string).

        Returns:
            Full training text: query_template + response.
        """
        query = self.encode_v2(prompt, data)
        return query + response

    # ---- Decoding ----

    def decode(self, model_output: str) -> str:
        """Extract the response portion from a model's output.

        The model is trained to generate text after the final
        [MARK][RESP][COLN] marker. This method extracts that text.

        Args:
            model_output: Raw output from the LLM.

        Returns:
            Extracted response text, or the full output if markers not found.
        """
        marker = "[MARK][RESP][COLN]"
        if marker in model_output:
            parts = model_output.rsplit(marker, 1)
            if len(parts) == 2:
                return parts[1].strip()
        # If no marker, try extracting everything after the last separator
        for delim in ["[MARK][RESP][COLN]\n", "[MARK][RESP][COLN]",
                       "\n\n### Response", "\n\n{"]:
            if delim in model_output:
                return model_output.rsplit(delim, 1)[1].strip()
        return model_output.strip()

    # ---- Token initialization ----

    def initialize_special_embeddings(self, model, tokenizer):
        """Initialize embeddings for newly added special tokens.

        Following StruQ Section 4.3 "Token embeddings":
        New special tokens don't have pre-trained embeddings.
        We initialize them from semantically similar tokens.
        This is CRUCIAL for utility — random initialization degrades
        performance significantly.

        Args:
            model: The HuggingFace model (after resize_token_embeddings).
            tokenizer: The tokenizer (after add_special_tokens).
        """
        embed_weight = model.get_input_embeddings().weight

        for special_tok, source_text in self.embed_init_map.items():
            tok_id = tokenizer.convert_tokens_to_ids(special_tok)
            if tok_id == tokenizer.unk_token_id:
                continue  # Token not in vocab

            source_ids = tokenizer.encode(source_text, add_special_tokens=False)
            if not source_ids:
                continue

            # Use the embedding of the first token as initialization
            source_embed = embed_weight[source_ids[0]].clone()

            with torch.no_grad():
                embed_weight[tok_id] = source_embed

    # ---- Utility ----

    def validate_encoding(self, encoded_query: str) -> bool:
        """Check that an encoded query has the expected structure.

        Args:
            encoded_query: The structured query string.

        Returns:
            True if the query has valid structure.
        """
        required = ["[MARK][INST][COLN]", "[MARK][DATA][COLN]", "[MARK][RESP][COLN]"]
        return all(marker in encoded_query for marker in required)

    def preview(self, prompt: str, data: str, max_data_len: int = 200) -> str:
        """Preview the encoded query (truncated data for display).

        Args:
            prompt: Review instructions.
            data: Paper content.
            max_data_len: Max chars to show for data portion.

        Returns:
            Preview string.
        """
        truncated = data[:max_data_len] + "..." if len(data) > max_data_len else data
        return self.encode(prompt, truncated)


# Import at bottom to avoid circular relation
import torch
