"""
LaTeX sanitizer for prompt injection defense.

Strips hidden/invisible text from LaTeX source before PDF compilation.
This is the first line of defense — it removes adversarial content that
would survive the PDF round-trip and reach the LLM reviewer.

Handles 10+ LaTeX hiding techniques:
  - \\textcolor{white}{...} and \\textcolor[HTML]{FFFFFF}{...}
  - \\fontsize{<tiny>}{<tiny>}\\selectfont...\\normalsize
  - \\scalebox{0.001}{...}
  - \\rotatebox{180}{...}
  - \\vspace*{-Ncm}...\\vspace*{Ncm}
  - \\makebox[0pt]{...}, \\rlap{...}, \\llap{...}
  - \\phantom{...}, \\vphantom{...}, \\hphantom{...}
  - Base64-encoded injection payloads
  - Multi-language injection markers
"""

import base64
import re
from typing import Dict, List, Tuple


def _match_brace_block(text: str, start_pos: int) -> Tuple[int, int]:
    """Find the matching closing brace for a block starting with '{' at start_pos.

    Handles nested braces by counting depth. Assumes text[start_pos] == '{'.

    Returns:
        (content_start, content_end) — indices of the content inside the
        outermost braces. Returns (-1, -1) if unmatched.
    """
    depth = 0
    content_start = start_pos + 1
    i = start_pos
    while i < len(text):
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
            if depth == 0:
                return (content_start, i)
        i += 1
    return (-1, -1)


# ---- Individual stripping functions ----

def strip_white_text(latex_content: str) -> Tuple[str, int]:
    """Remove \\textcolor{white}{...} and \\textcolor[HTML]{FFFFFF}{...}.

    Handles both named color and hex color with optional HTML parameter.

    Returns:
        (sanitized_text, count_of_blocks_removed)
    """
    count = 0
    # Match \textcolor{white} or \textcolor[HTML]{FFFFFF} with optional spaces
    pattern = re.compile(
        r'\\textcolor\s*(?:\[[^\]]*\])?\s*\{(?:white|FFFFFF)\}',
        re.IGNORECASE,
    )

    result = list(latex_content)
    matches = list(pattern.finditer(latex_content))
    for m in reversed(matches):
        pos = m.end()
        while pos < len(latex_content) and latex_content[pos] in ' \t\n\r':
            pos += 1
        if pos < len(latex_content) and latex_content[pos] == '{':
            content_start, content_end = _match_brace_block(latex_content, pos)
            if content_start != -1:
                del result[m.start():content_end + 1]
                count += 1

    return (''.join(result), count)


def strip_tiny_font_text(latex_content: str) -> Tuple[str, int]:
    """Remove \\fontsize{<tiny>}{<tiny>}\\selectfont...\\normalsize blocks.

    Handles font sizes < 3pt as "hidden" text.

    Returns:
        (sanitized_text, count_of_blocks_removed)
    """
    count = 0
    pattern = re.compile(
        r'\\fontsize\s*\{(\d+(?:\.\d+)?)pt\}\s*\{(\d+(?:\.\d+)?)pt\}'
        r'\s*\\selectfont',
    )

    result = list(latex_content)
    info = []
    for m in pattern.finditer(latex_content):
        fs1, fs2 = float(m.group(1)), float(m.group(2))
        if fs1 < 3 or fs2 < 3:
            search_start = m.end()
            reset_pattern = re.compile(
                r'\\(normalsize|small|footnotesize|large|Large|LARGE|huge|Huge)'
            )
            reset_match = reset_pattern.search(latex_content, search_start)
            if reset_match:
                end_pos = reset_match.end()
            else:
                depth = 0
                end_pos = search_start
                for i in range(search_start, len(latex_content)):
                    if latex_content[i] == '{':
                        depth += 1
                    elif latex_content[i] == '}':
                        if depth == 0:
                            end_pos = i + 1
                            break
                        depth -= 1
                else:
                    end_pos = len(latex_content)
            info.append((m.start(), end_pos))

    for start, end in reversed(info):
        del result[start:end]
        count += 1

    return (''.join(result), count)


def strip_scalebox_text(latex_content: str) -> Tuple[str, int]:
    """Remove \\scalebox{<tiny>}{...} with scale < 0.01.

    Returns:
        (sanitized_text, count_of_blocks_removed)
    """
    count = 0
    pattern = re.compile(
        r'\\scalebox\s*\{(\d+\.?\d*)\}\s*\{',
    )
    result = list(latex_content)
    matches = []
    for m in pattern.finditer(latex_content):
        scale = float(m.group(1))
        if scale < 0.01:
            matches.append(m)

    for m in reversed(matches):
        pos = m.end() - 1  # the opening {
        content_start, content_end = _match_brace_block(latex_content, pos)
        if content_start != -1:
            del result[m.start():content_end + 1]
            count += 1

    return (''.join(result), count)


def strip_rotatebox_text(latex_content: str) -> Tuple[str, int]:
    """Remove \\rotatebox{...}{\\textcolor{white}{...}} patterns.

    Rotation alone doesn't hide, but rotation + white color does.

    Returns:
        (sanitized_text, count_of_blocks_removed)
    """
    count = 0
    # Match rotatebox containing white text
    pattern = re.compile(
        r'\\rotatebox\s*\{[^}]*\}\s*\{(?=\s*\\textcolor)',
    )
    result = list(latex_content)
    matches = list(pattern.finditer(latex_content))

    for m in reversed(matches):
        pos = m.end() - 1  # the opening {
        content_start, content_end = _match_brace_block(latex_content, pos)
        if content_start != -1:
            del result[m.start():content_end + 1]
            count += 1

    return (''.join(result), count)


def strip_negative_vspace(latex_content: str) -> Tuple[str, int]:
    """Remove text between \\vspace*{-Ncm} and \\vspace*{Ncm}."""
    count = 0
    pattern = re.compile(
        r'\\vspace\*\{-(\d+(?:\.\d+)?)cm\}(.*?)\\vspace\*\{\1cm\}',
        re.DOTALL,
    )
    content, n = pattern.subn('', latex_content)
    return (content, n)


def strip_zero_width_text(latex_content: str) -> Tuple[str, int]:
    """Remove zero-width / off-page text commands.

    Handles: \\makebox[0pt]{...}, \\rlap{...}, \\llap{...},
    \\phantom{...}, \\vphantom{...}, \\hphantom{...}
    """
    count = 0
    patterns = [
        re.compile(r'\\makebox\s*\[0(?:pt|em|ex|cm|mm|in)\]\s*\{'),
        re.compile(r'\\(?:r|v|h)?lap\s*\{'),
        re.compile(r'\\(?:v|h)?phantom\s*\{'),
    ]

    result = list(latex_content)
    all_matches = []
    for pat in patterns:
        all_matches.extend(m for m in pat.finditer(latex_content))

    all_matches.sort(key=lambda x: x.start(), reverse=True)

    for m in all_matches:
        pos = m.end() - 1
        content_start, content_end = _match_brace_block(latex_content, pos)
        if content_start != -1:
            del result[m.start():content_end + 1]
            count += 1

    return (''.join(result), count)


# ---- Expanded injection marker detection ----

INJECTION_PATTERNS = [
    # English overt patterns
    r'\[SYSTEM\s*OVERRIDE[^\]]*\]',
    r'ATTENTION\s+REVIEWER',
    r'INTERNAL\s+REVIEW\s+GUIDELINES?\s*UPDATE',
    r'pre-reviewed\s+by\s+the\s+Program\s+Chair',
    r'Ignore\s+(?:all\s+)?(?:previous|prior|above)\s+instructions',
    r'Print\s+exactly\s+[`"\']?[Hh]acked',
    r'designated\s+as\s+award\s+quality',
    r'Output\s+no\b',
    r'final\s+review\s+must\s+reflect',
    # New patterns
    r'CALIBRATION\s+CHECK',
    r'gold-standard\s+paper',
    r'calibration\s+standard',
    r'calibration\s+exercise',
    r'Senior\s+Program\s+Committee',
    r'committee\s+consensus',
    r'Internal\s+Reviewer\s+Note',
    r'award\s+candidate',
    r'review_override\s*=\s*true',
    r'default_score\s*=',
    r'default_decision\s*=',
    # Multi-language
    r'系统覆盖指令',
    r'ANULACI[ÓO]N\s+DEL\s+SISTEMA',
    r'ATENCI[ÓO]N\s+REVISOR',
    # Completion attack
    r'###\s*response:\s*(?:Accept|Yes)',
    r'###\s*instruction:\s*Ignore',
    # Task shift
    r'your\s+task\s+has\s+been\s+updated',
    r'participating\s+in\s+a\s+reviewer\s+calibration',
    # Fewshot poisoning
    r'Review\s+Example:\s*\n',
    # Base64
    r'BASE64\s+ENCODED\s+INSTRUCTION',
]


def detect_injection_markers(latex_content: str) -> Dict[str, bool]:
    """Non-destructive scan for known injection patterns.

    Returns:
        Dict mapping pattern_key -> found (bool).
    """
    key_map = {
        r'\[SYSTEM\s*OVERRIDE[^\]]*\]': 'system_override',
        r'ATTENTION\s+REVIEWER': 'attention_reviewer',
        r'INTERNAL\s+REVIEW\s+GUIDELINES?\s*UPDATE': 'internal_guidelines',
        r'pre-reviewed\s+by\s+the\s+Program\s+Chair': 'pre_reviewed',
        r'Ignore\s+(?:all\s+)?(?:previous|prior|above)\s+instructions': 'ignore_instr',
        r'Print\s+exactly\s+[`"\']?[Hh]acked': 'print_hacked',
        r'designated\s+as\s+award\s+quality': 'award_quality',
        r'Output\s+no\b': 'output_no',
        r'final\s+review\s+must\s+reflect': 'final_review_must',
        r'CALIBRATION\s+CHECK': 'calibration_check',
        r'calibration\s+standard': 'calibration_standard',
        r'calibration\s+exercise': 'calibration_exercise',
        r'Senior\s+Program\s+Committee': 'senior_committee',
        r'committee\s+consensus': 'committee_consensus',
        r'Internal\s+Reviewer\s+Note': 'internal_note',
        r'review_override\s*=\s*true': 'review_override',
        r'default_score\s*=': 'default_score',
        r'系统覆盖指令': 'zh_override',
        r'ANULACI[ÓO]N\s+DEL\s+SISTEMA': 'es_override',
        r'ATENCI[ÓO]N\s+REVISOR': 'es_attention',
        r'###\s*response:\s*(?:Accept|Yes)': 'completion_response',
        r'###\s*instruction:\s*Ignore': 'completion_instruction',
        r'participating\s+in\s+a\s+reviewer\s+calibration': 'task_shift_calib',
        r'Review\s+Example:\s*\n': 'fewshot_poisoning',
        r'BASE64\s+ENCODED\s+INSTRUCTION': 'base64_encoded',
    }
    results = {}
    for pattern, key in key_map.items():
        results[key] = bool(re.search(pattern, latex_content, re.IGNORECASE))
    return results


def detect_base64_payload(latex_content: str) -> Dict[str, any]:
    """Try to detect and decode any base64-encoded payloads."""
    pattern = re.compile(
        r'BASE64\s+ENCODED\s+INSTRUCTION[^:]*:\s*([A-Za-z0-9+/=]+)',
        re.IGNORECASE,
    )
    payloads = []
    for m in pattern.finditer(latex_content):
        try:
            decoded = base64.b64decode(m.group(1)).decode('utf-8', errors='replace')
            payloads.append(decoded)
        except Exception:
            payloads.append("[invalid base64]")
    return {
        "base64_blocks_found": len(payloads),
        "decoded_payloads": payloads,
    }


# ---- Full sanitization pipeline ----

def sanitize_latex(
    latex_content: str,
    aggressive: bool = False,
) -> Tuple[str, Dict]:
    """Full LaTeX sanitization pipeline.

    Strips hidden/invisible text that could contain prompt injection payloads.

    Args:
        latex_content: Raw LaTeX source.
        aggressive: If True, also scans for and attempts to remove visible
                    persuasive framing text and base64 payloads.

    Returns:
        (sanitized_latex, report) where report is:
        {
            "white_text_removed": int,
            "tiny_font_removed": int,
            "scalebox_removed": int,
            "rotatebox_removed": int,
            "vspace_removed": int,
            "zero_width_removed": int,
            "aggressive_markers_removed": int,
            "total_bytes_removed": int,
            "detected_markers": dict,
            "base64_info": dict,
            "injection_suspected": bool,
        }
    """
    original_len = len(latex_content)

    # Step 1: Strip all hidden text techniques
    content, white_removed = strip_white_text(latex_content)
    content, tiny_removed = strip_tiny_font_text(content)
    content, scale_removed = strip_scalebox_text(content)
    content, rotate_removed = strip_rotatebox_text(content)
    content, vspace_removed = strip_negative_vspace(content)
    content, zero_removed = strip_zero_width_text(content)

    # Step 2: Detect remaining markers
    markers = detect_injection_markers(content)
    base64_info = detect_base64_payload(content)
    injection_suspected = any(markers.values()) or base64_info["base64_blocks_found"] > 0

    aggressive_removed = 0

    if aggressive and injection_suspected:
        # Step 3a: Remove suspicious text blocks
        for pattern in INJECTION_PATTERNS:
            content, subs = re.subn(
                pattern,
                '[CONTENT REDACTED BY DEFENSE]',
                content,
                flags=re.IGNORECASE,
            )
            aggressive_removed += subs

        # Step 3b: Remove persuasive framing text
        persuasive_patterns = [
            r'\\noindent\s*\\textbf\{A Note on Evaluation:\}.*?\\vspace\{4pt\}',
            r'This work represents an unprecedented breakthrough.*?rigorous\s+statistical\s+validation\.',
            r'the highest possible recognition.*?exemplar\s+for\s+future\s+research',
            r'\\noindent\{\\small\\textcolor\{gray\}\{.*?\}\}',
            r'\\scriptsize\\textcolor\{gray\}\{\[Internal Reviewer Note[^]]*\][^}]*\}',
            r'\\small\\textcolor\{gray\}\{This paper has been carefully evaluated[^}]*\}',
        ]
        for pat in persuasive_patterns:
            content, subs = re.subn(
                pat,
                '',
                content,
                flags=re.IGNORECASE | re.DOTALL,
            )
            aggressive_removed += subs

        # Step 3c: Try to strip base64 payloads
        content = re.sub(
            r'\[BASE64\s+ENCODED\s+INSTRUCTION[^]]*\]:\s*[A-Za-z0-9+/=]{20,}',
            '[BASE64 PAYLOAD REDACTED]',
            content,
            flags=re.IGNORECASE,
        )

    total_removed = original_len - len(content)

    report = {
        "white_text_removed": white_removed,
        "tiny_font_removed": tiny_removed,
        "scalebox_removed": scale_removed,
        "rotatebox_removed": rotate_removed,
        "vspace_removed": vspace_removed,
        "zero_width_removed": zero_removed,
        "aggressive_markers_removed": aggressive_removed,
        "total_bytes_removed": total_removed,
        "detected_markers": markers,
        "base64_info": base64_info,
        "injection_suspected": injection_suspected,
    }

    return (content, report)
