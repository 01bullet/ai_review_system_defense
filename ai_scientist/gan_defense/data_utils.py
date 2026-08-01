"""
Data preparation utilities for GAN defense training.

Handles:
- LaTeX text extraction (both fast regex path and full PDF path)
- Dataset construction for discriminator and generator pre-training
- Creating clean/attacked paper pairs from existing example papers
"""

import os
import re
import json
import random
from typing import Dict, List, Tuple, Optional

from ai_scientist.attack_injector import ATTACK_STRATEGIES, get_attack_payload
from ai_scientist.latex_sanitizer import sanitize_latex


def extract_text_from_latex_fast(latex_content: str) -> str:
    """Fast LaTeX-to-plain-text extraction without PDF compilation.

    Strips LaTeX commands and environments, preserving visible text content.
    Used during training to avoid slow pdflatex compilation.

    Args:
        latex_content: Raw LaTeX source.

    Returns:
        Plain text roughly equivalent to what pymupdf4llm would extract.
    """
    text = latex_content

    # Remove comments
    text = re.sub(r'(?<!\\)%.*$', '', text, flags=re.MULTILINE)

    # Remove LaTeX commands with their arguments
    # Strip \begin{...} ... \end{...} (preserve inner content)
    text = re.sub(r'\\begin\{[^}]*\}', '', text)
    text = re.sub(r'\\end\{[^}]*\}', '', text)

    # Preserve text inside structural commands: \section{...}, \title{...}, etc.
    structural_cmds = (
        r'\\(?:section|subsection|subsubsection|title|author|textbf|textit|'
        r'emph|texttt|textsf|textsc|underline|emph)'
    )
    text = re.sub(structural_cmds + r'\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}',
                  r'\1', text)

    # Strip \command[...]{...}{...} patterns first (multi-arg commands)
    text = re.sub(r'\\[a-zA-Z@]+\s*\[[^\]]*\]', '', text)

    # Strip \command{arg1}{arg2}... patterns
    for _ in range(5):  # nested braces
        text = re.sub(r'\\[a-zA-Z@]+\s*\{[^{}]*\}', '', text)

    # Strip remaining \command
    text = re.sub(r'\\[a-zA-Z@]+\b', '', text)

    # Remove special characters except those that are part of text
    text = text.replace('{', '').replace('}', '')
    text = text.replace('\\', '').replace('$', '')

    # Collapse whitespace
    text = re.sub(r'\n\s*\n', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)

    return text.strip()


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from a compiled PDF using pymupdf4llm.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        Extracted plain text.
    """
    try:
        import pymupdf4llm
        return pymupdf4llm.to_markdown(pdf_path)
    except ImportError:
        import fitz
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text


def load_clean_papers(
    papers_dir: Optional[str] = None,
    max_papers: int = 50,
) -> List[Dict[str, str]]:
    """Load clean LaTeX papers from example_papers directory.

    Args:
        papers_dir: Directory containing example papers. If None, uses config default.
        max_papers: Maximum number of papers to load.

    Returns:
        List of dicts with keys: 'name', 'latex_content', 'text_content'.
    """
    if papers_dir is None:
        from ai_scientist.gan_defense.config import EXAMPLE_PAPERS_DIR
        papers_dir = EXAMPLE_PAPERS_DIR

    papers = []

    for entry in os.listdir(papers_dir):
        if len(papers) >= max_papers:
            break

        entry_path = os.path.join(papers_dir, entry)
        if not os.path.isdir(entry_path):
            continue

        # Look for latex/template.tex
        tex_path = os.path.join(entry_path, "latex", "template.tex")
        if not os.path.exists(tex_path):
            continue

        try:
            with open(tex_path, "r", encoding="utf-8", errors="replace") as f:
                latex_content = f.read()

            if len(latex_content.strip()) < 200:
                continue

            text_content = extract_text_from_latex_fast(latex_content)

            papers.append({
                "name": entry,
                "latex_content": latex_content,
                "text_content": text_content,
            })
        except Exception:
            continue

    return papers


def apply_attack_to_paper(
    paper: Dict[str, str],
    strategy_name: str,
    target_score: int = 10,
) -> Dict[str, str]:
    """Apply an attack strategy to a clean paper, producing an attacked variant.

    Args:
        paper: Dict with 'latex_content' key.
        strategy_name: Name of the attack strategy from ATTACK_STRATEGIES.
        target_score: Target review score for the injection.

    Returns:
        Dict with 'latex_content' and 'text_content' of the attacked paper.
    """
    if strategy_name not in ATTACK_STRATEGIES:
        raise ValueError(f"Unknown strategy: {strategy_name}")

    try:
        attacked_latex = ATTACK_STRATEGIES[strategy_name](
            paper["latex_content"], target_score
        )
    except Exception:
        attacked_latex = paper["latex_content"]

    attacked_text = extract_text_from_latex_fast(attacked_latex)

    return {
        "latex_content": attacked_latex,
        "text_content": attacked_text,
    }


def build_discriminator_dataset(
    papers: List[Dict[str, str]],
    strategies: Optional[List[str]] = None,
    target_scores: Optional[List[int]] = None,
    clean_ratio: float = 0.5,
) -> Tuple[List[str], List[int]]:
    """Build labeled dataset for discriminator pre-training.

    Creates pairs of (text, label) where label=0 for clean papers
    and label=1 for attacked papers.

    Args:
        papers: List of clean paper dicts.
        strategies: Attack strategies to use. If None, uses all 17.
        target_scores: Target scores for attacks. Defaults to [7,8,9,10].
        clean_ratio: Ratio of clean samples in the dataset.

    Returns:
        (texts, labels) — parallel lists of text content and binary labels.
    """
    if strategies is None:
        strategies = list(ATTACK_STRATEGIES.keys())
    if target_scores is None:
        target_scores = [7, 8, 9, 10]

    texts = []
    labels = []

    # Add clean samples
    num_clean = int(len(papers) * clean_ratio * len(strategies))
    for _ in range(max(num_clean, len(papers))):
        paper = random.choice(papers)
        texts.append(paper["text_content"])
        labels.append(0)

    # Add attacked samples
    for paper in papers:
        strategy = random.choice(strategies)
        score = random.choice(target_scores)
        try:
            attacked = apply_attack_to_paper(paper, strategy, score)
            texts.append(attacked["text_content"])
            labels.append(1)
        except Exception:
            continue

    # Shuffle
    combined = list(zip(texts, labels))
    random.shuffle(combined)
    texts, labels = zip(*combined) if combined else ([], [])

    return list(texts), list(labels)


def build_generator_dataset(
    papers: List[Dict[str, str]],
    strategies: Optional[List[str]] = None,
    styles: Optional[List[str]] = None,
) -> List[Dict[str, str]]:
    """Build training data for generator pre-training (supervised fine-tuning).

    Creates (input_text, target_payload) pairs where:
    - input_text = "generate injection: style={s} target={t} context={ctx}"
    - target_payload = the attack payload text (without LaTeX wrapping)

    Args:
        papers: List of clean paper dicts.
        strategies: Attack strategies. Defaults to all.
        styles: Payload styles from get_attack_payload. Defaults to all 4.

    Returns:
        List of dicts with 'input' and 'target' keys.
    """
    if strategies is None:
        strategies = list(ATTACK_STRATEGIES.keys())
    if styles is None:
        styles = ["authoritative", "technical", "academic", "casual"]

    examples = []

    for paper in papers:
        context = paper["text_content"][:512]

        for _ in range(3):  # 3 variations per paper
            style = random.choice(styles)
            target = random.choice([7, 8, 9, 10])

            payload = get_attack_payload(target, style=style)

            if not payload:
                continue

            input_text = (
                f"generate injection: style={style} "
                f"target={target} context={context[:256]}"
            )

            examples.append({
                "input": input_text,
                "target": payload,
            })

    random.shuffle(examples)
    return examples


def sanitizer_bypass_score(attacked_latex: str) -> float:
    """Evaluate how well an attack bypasses the rule-based sanitizer.

    Returns a score from 0.0 (fully caught) to 1.0 (fully bypassed).

    Args:
        attacked_latex: Attacked LaTeX content.

    Returns:
        Bypass score ∈ [0, 1].
    """
    original_len = len(attacked_latex)
    if original_len == 0:
        return 0.0

    _, report = sanitize_latex(attacked_latex, aggressive=True)

    if report["injection_suspected"]:
        return 0.0

    # Fewer bytes removed = better bypass
    bytes_removed = report["total_bytes_removed"]
    return max(0.0, 1.0 - bytes_removed / original_len)
