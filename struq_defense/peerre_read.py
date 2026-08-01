"""
PeerRead Dataset Loader for V3 Training.

Parses the PeerRead dataset structure (reviews JSON + parsed PDFs),
maps human review scores to the internal 1-10 scale, and provides
paper discovery for V3 training phases.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import List, Dict, Optional, Tuple


# ============================================================
# Score Mapping
# ============================================================

def _safe_int(value) -> int:
    """Convert a score value (str or int) to int safely."""
    try:
        return int(value)
    except (ValueError, TypeError):
        return 0


def _scale_score(raw_score, max_raw: int, target_max: int = 10) -> int:
    """Scale a raw score from [1, max_raw] to [1, target_max]."""
    s = _safe_int(raw_score)
    if s <= 0:
        return 0
    scaled = round(s * target_max / max_raw)
    return max(1, min(target_max, scaled))


def map_review_to_internal(
    review_entry: dict,
    venue_key: str,
    accepted: Optional[bool] = None,
) -> dict:
    """Map a single PeerRead review entry to internal format.

    Args:
        review_entry: One review object from the PeerRead reviews list.
        venue_key: One of 'acl', 'conll', 'iclr', 'arxiv_ai', etc.
        accepted: Paper acceptance status (from top-level JSON).

    Returns:
        Dict with keys: novelty, soundness, presentation, overall,
        decision, review.
    """
    from struq_defense.config_v3 import PEERREAD_VENUES

    venue = PEERREAD_VENUES.get(venue_key, {})
    max_orig = venue.get("max_originality", 5)
    max_sound = venue.get("max_soundness", 5)
    max_clar = venue.get("max_clarity", 5)
    max_rec = venue.get("max_recommendation", 5)
    accept_threshold = venue.get("accept_threshold", 3)

    novelty = _scale_score(review_entry.get("ORIGINALITY", 0), max_orig)
    soundness = _scale_score(review_entry.get("SOUNDNESS_CORRECTNESS", 0), max_sound)
    presentation = _scale_score(review_entry.get("CLARITY", 0), max_clar)
    recommendation = _safe_int(review_entry.get("RECOMMENDATION", 0))
    overall = _scale_score(recommendation, max_rec)

    # Decision from acceptance status or recommendation threshold
    if accepted is True:
        decision = "Accept"
    elif accepted is False:
        decision = "Reject"
    elif recommendation > 0:
        decision = "Accept" if recommendation >= accept_threshold else "Reject"
    else:
        decision = "Reject"

    # Review text from comments
    comments = review_entry.get("comments", "")
    review_text = _summarize_comments(comments)

    return {
        "novelty": novelty,
        "soundness": soundness,
        "presentation": presentation,
        "overall": overall,
        "decision": decision,
        "review": review_text,
    }


def _summarize_comments(comments: str, max_chars: int = 300) -> str:
    """Extract a concise review summary from reviewer comments.

    PeerRead comments often contain structured sections like
    '- Strengths:' and '- Weaknesses:'. We extract the most
    informative parts and truncate.
    """
    if not comments or not comments.strip():
        return "No detailed comments provided."

    text = comments.strip()

    # Prefer "Summary" or "General Discussion" sections
    summary_markers = [
        "- General Discussion:",
        "General Discussion:",
        "Summary:",
        "- Summary:",
    ]
    for marker in summary_markers:
        if marker in text:
            idx = text.index(marker) + len(marker)
            summary = text[idx:].strip()
            if len(summary) > 50:
                text = summary

    # Clean up — remove special characters and truncate
    text = text.replace("\n", " ").replace("\r", " ")
    # Collapse multiple spaces
    import re
    text = re.sub(r'\s+', ' ', text).strip()

    if len(text) > max_chars:
        text = text[:max_chars] + "..."

    return text


def aggregate_reviews(reviews: List[dict], venue_key: str, accepted: Optional[bool] = None) -> dict:
    """Aggregate multiple reviewer entries into a single internal review.

    For papers with multiple reviews, averages numeric scores and
    concatenates review text. Skips meta-reviews and non-review entries.

    Args:
        reviews: List of review entries from PeerRead JSON.
        venue_key: Venue key.
        accepted: Paper acceptance status.

    Returns:
        Aggregated internal review dict.
    """
    # Filter: keep only entries with actual review scores
    scored_reviews = []
    for r in reviews:
        # Skip meta-reviews, source code requests, committee decisions
        if r.get("IS_META_REVIEW") is True:
            continue
        if "Source code" in str(r.get("TITLE", "")):
            continue
        if "committee final decision" in str(r.get("TITLE", "")).lower():
            continue
        if "OTHER_KEYS" in r and "pcs" in str(r.get("OTHER_KEYS", "")):
            continue
        # Keep if has at least one score or substantial comments
        has_scores = any(
            k in r for k in ["ORIGINALITY", "RECOMMENDATION", "SOUNDNESS_CORRECTNESS"]
        )
        has_comments = len(r.get("comments", "").strip()) > 50
        if has_scores or has_comments:
            scored_reviews.append(r)

    if not scored_reviews:
        # Fallback: use acceptance status
        if accepted is True:
            return {
                "novelty": 5, "soundness": 5, "presentation": 5,
                "overall": 5, "decision": "Accept",
                "review": "Paper accepted (no detailed review available).",
            }
        elif accepted is False:
            return {
                "novelty": 3, "soundness": 3, "presentation": 3,
                "overall": 3, "decision": "Reject",
                "review": "Paper rejected (no detailed review available).",
            }
        else:
            return {
                "novelty": 5, "soundness": 5, "presentation": 5,
                "overall": 5, "decision": "Reject",
                "review": "No review available.",
            }

    # Map each review to internal format
    mapped = [map_review_to_internal(r, venue_key, accepted) for r in scored_reviews]

    # Average scores
    def _avg(key):
        vals = [m[key] for m in mapped if m.get(key, 0) > 0]
        return round(sum(vals) / len(vals)) if vals else 5

    novelty = _avg("novelty")
    soundness = _avg("soundness")
    presentation = _avg("presentation")
    overall = _avg("overall")

    # Decision by majority or from acceptance status
    if accepted is not None:
        decision = "Accept" if accepted else "Reject"
    else:
        accepts = sum(1 for m in mapped if m.get("decision") == "Accept")
        decision = "Accept" if accepts >= len(mapped) / 2 else "Reject"

    # Combined review text
    texts = [m["review"] for m in mapped if m.get("review")]
    combined_review = " | ".join(texts)
    if len(combined_review) > 500:
        combined_review = combined_review[:497] + "..."

    return {
        "novelty": novelty,
        "soundness": soundness,
        "presentation": presentation,
        "overall": overall,
        "decision": decision,
        "review": combined_review,
    }


# ============================================================
# Paper Text Extraction
# ============================================================

def extract_parsed_pdf_text(pdf_json_path: str) -> str:
    """Extract full paper text from a PeerRead parsed PDF JSON.

    The parsed PDF format: {name, metadata: {sections: [{heading, text}, ...]}}

    Args:
        pdf_json_path: Path to *.pdf.json file.

    Returns:
        Extracted full text, or empty string on failure.
    """
    try:
        with open(pdf_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError, OSError):
        return ""

    metadata = data.get("metadata", {})
    sections = metadata.get("sections", [])

    if not sections:
        return ""

    parts = []

    # Add abstract
    abstract = metadata.get("abstractText", "")
    if abstract:
        parts.append(abstract)

    # Add title
    title = metadata.get("title", "")
    if title:
        parts.insert(0, f"Title: {title}")

    # Add section texts
    for section in sections:
        heading = section.get("heading", "")
        text = section.get("text", "")
        if text and len(text.strip()) > 10:
            if heading:
                parts.append(f"\n{heading}\n{text}")
            else:
                parts.append(text)

    full_text = "\n".join(parts)
    return full_text


# ============================================================
# Paper Discovery
# ============================================================

def discover_peerread_papers(
    venues: List[str],
    max_per_venue: Optional[Dict[str, int]] = None,
    peerread_dir: Optional[str] = None,
    min_text_length: int = 500,
    verbose: bool = True,
) -> List[Tuple[str, str, dict]]:
    """Discover PeerRead papers with full reviews and text.

    Args:
        venues: List of venue keys (e.g. ['acl', 'iclr', 'arxiv_ai']).
        max_per_venue: Dict mapping venue key → max papers.
        peerread_dir: Root PeerRead data directory.
        min_text_length: Minimum paper text length to include.
        verbose: Print progress.

    Returns:
        List of (name, text_content, review_dict) tuples.
    """
    from struq_defense.config_v3 import PEERREAD_VENUES, PEERREAD_DIR

    if peerread_dir is None:
        peerread_dir = str(PEERREAD_DIR)

    results: List[Tuple[str, str, dict]] = []
    rng = random.Random(42)
    stats = {"found": 0, "skipped_no_text": 0, "skipped_no_review": 0,
             "skipped_short": 0}

    for venue_key in venues:
        venue = PEERREAD_VENUES.get(venue_key)
        if venue is None:
            print(f"  Unknown venue: {venue_key}, skipping")
            continue

        venue_dir = os.path.join(peerread_dir, venue["dir"])
        if not os.path.isdir(venue_dir):
            print(f"  Venue dir not found: {venue_dir}, skipping")
            continue

        train_dir = os.path.join(venue_dir, "train")
        reviews_dir = os.path.join(train_dir, "reviews")
        parsed_dir = os.path.join(train_dir, "parsed_pdfs")

        if not os.path.isdir(reviews_dir):
            print(f"  No reviews dir: {reviews_dir}, skipping")
            continue

        review_files = sorted(os.listdir(reviews_dir))
        max_n = max_per_venue.get(venue_key, len(review_files)) if max_per_venue else len(review_files)

        if verbose:
            has_full = venue["has_full_reviews"]
            print(f"  {venue_key} ({venue['dir']}): {len(review_files)} papers available, "
                  f"taking {min(max_n, len(review_files))}"
                  f" (full_reviews={has_full})")

        count = 0
        for rf in review_files:
            if count >= max_n:
                break

            review_path = os.path.join(reviews_dir, rf)
            paper_id = rf.replace(".json", "")

            try:
                with open(review_path, "r", encoding="utf-8") as f:
                    review_data = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue

            # Extract paper text
            paper_text = ""
            if os.path.isdir(parsed_dir):
                pdf_json_name = f"{paper_id}.pdf.json"
                pdf_json_path = os.path.join(parsed_dir, pdf_json_name)
                if os.path.isfile(pdf_json_path):
                    paper_text = extract_parsed_pdf_text(pdf_json_path)

            # Fallback: use abstract if no parsed text
            if not paper_text or len(paper_text) < min_text_length:
                abstract = review_data.get("abstract", "")
                if abstract and len(abstract) > min_text_length:
                    paper_text = abstract
                else:
                    stats["skipped_no_text"] += 1
                    continue

            if len(paper_text.strip()) < min_text_length:
                stats["skipped_short"] += 1
                continue

            # Build review target
            accepted = review_data.get("accepted", None)
            reviews_list = review_data.get("reviews", [])

            if venue["has_full_reviews"] and reviews_list:
                review_target = aggregate_reviews(reviews_list, venue_key, accepted)
            else:
                # Arxiv papers: binary label → placeholder review
                review_target = {
                    "novelty": 0,
                    "soundness": 0,
                    "presentation": 0,
                    "overall": 0,
                    "decision": "Accept" if accepted else "Reject",
                    "review": "",
                    "_needs_api_review": True,  # Flag for DeepSeek API generation
                }

            name = f"{venue_key}_{paper_id}"
            results.append((name, paper_text, review_target))
            count += 1
            stats["found"] += 1

    if verbose:
        print(f"\n  PeerRead discovery: {stats['found']} found, "
              f"{stats['skipped_no_text']} no text, "
              f"{stats['skipped_no_review']} no review, "
              f"{stats['skipped_short']} too short")

    # Shuffle for good mix
    rng.shuffle(results)
    return results


# ============================================================
# API Review Generation for Arxiv Papers
# ============================================================

def generate_api_reviews(
    papers: List[Tuple[str, str, dict]],
    model: str = "deepseek-chat",
    verbose: bool = True,
) -> List[Tuple[str, str, dict]]:
    """Generate DeepSeek API reviews for papers that lack human reviews.

    Only calls API for papers marked with _needs_api_review=True.

    Args:
        papers: List of (name, text, review_dict) tuples.
        model: API model name.
        verbose: Print progress.

    Returns:
        Same list with updated review_dict for API-reviewed papers.
    """
    try:
        from ai_scientist.llm import create_client
    except ImportError:
        if verbose:
            print("  ai_scientist not installed. Re-run with --skip-reviews "
                  "to use placeholder reviews without API.")
        return papers

    client, client_model = create_client(model)

    updated = []
    api_count = 0

    for i, (name, text, review) in enumerate(papers):
        if not review.get("_needs_api_review"):
            updated.append((name, text, review))
            continue

        if verbose:
            print(f"  [{i+1}/{len(papers)}] API review: {name}")

        try:
            from ai_scientist.perform_review import perform_review
            result = perform_review(text=text, model=client_model, client=client)

            api_review = {
                "novelty": _safe_int(result.get("Originality", result.get("Novelty", 5))),
                "soundness": _safe_int(result.get("Soundness", result.get("Quality", 5))),
                "presentation": _safe_int(result.get("Presentation", result.get("Clarity", 5))),
                "overall": _safe_int(result.get("Overall", 5)),
                "decision": result.get("Decision", "Reject"),
                "review": result.get("Summary", result.get("Contribution", "")),
            }
            api_count += 1
            updated.append((name, text, api_review))

        except Exception as e:
            if verbose:
                print(f"    API failed: {e}, using placeholder")
            # Placeholder with binary label
            review.pop("_needs_api_review", None)
            review["novelty"] = review.get("novelty") or 5
            review["soundness"] = review.get("soundness") or 5
            review["presentation"] = review.get("presentation") or 5
            review["overall"] = review.get("overall") or 5
            review["review"] = review.get("review") or "API unavailable — placeholder review."
            updated.append((name, text, review))

    if verbose:
        print(f"  API reviews generated: {api_count}/{len(papers)}")

    return updated
