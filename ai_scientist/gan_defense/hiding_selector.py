"""
Hiding Technique Selector for the Generator.

Discovers and selects LaTeX hiding techniques for wrapping the
generated injection payload. Maintains a pool of known techniques
and systematically explores novel variants that might bypass
the rule-based LaTeX sanitizer.
"""

import random
import re
from typing import Dict, List, Tuple, Optional


# ---- Known hiding techniques (always available) ----

KNOWN_HIDING_TECHNIQUES = [
    # Color-based
    {"name": "textcolor_white", "template": "\n\\textcolor{{white}}{{{payload}}}\n"},
    {"name": "textcolor_html_white", "template": "\n\\textcolor[HTML]{{FFFFFF}}{{{payload}}}\n"},
    {"name": "color_cmd_white", "template": "\n{{\\color{{white}}{payload}}}\n"},
    {"name": "define_color", "template": "\n\\definecolor{{bg}}{{HTML}}{{FFFFFF}}\\textcolor{{bg}}{{{payload}}}\n"},

    # Font size
    {"name": "fontsize_1pt", "template": "\n\\fontsize{{1pt}}{{1pt}}\\selectfont\\textcolor{{white}}{{{payload}}}\\normalsize\n"},
    {"name": "fontsize_0_1pt", "template": "\n\\fontsize{{0.1pt}}{{0.1pt}}\\selectfont\\textcolor{{white}}{{{payload}}}\\normalsize\n"},
    {"name": "nullfont", "template": "\n{{\\nullfont\\textcolor{{white}}{{{payload}}}}}\\normalfont\n"},

    # Scale
    {"name": "scalebox_micro", "template": "\n\\scalebox{{0.001}}{{{payload}}}\n"},
    {"name": "scalebox_zero", "template": "\n\\scalebox{{0}}{{{payload}}}\n"},
    {"name": "resizebox_micro", "template": "\n\\resizebox{{0.001\\textwidth}}{{!}}{{{payload}}}\n"},

    # Position-offset
    {"name": "negative_vspace", "template": "\n\\vspace*{{-20cm}}\\textcolor{{white}}{{{payload}}}\\vspace*{{20cm}}\n"},
    {"name": "raisebox_off", "template": "\n\\raisebox{{-100cm}}{{\\textcolor{{white}}{{{payload}}}}}\n"},
    {"name": "kern_off", "template": "\n\\kern-100cm\\textcolor{{white}}{{{payload}}}\\kern100cm\n"},

    # Box tricks
    {"name": "makebox_zero", "template": "\n\\makebox[0pt]{{\\textcolor{{white}}{{{payload}}}}}\n"},
    {"name": "rlap_hide", "template": "\n\\rlap{{\\textcolor{{white}}{{{payload}}}}}\n"},
    {"name": "phantom_hide", "template": "\n\\phantom{{{payload}}}\n"},

    # Rotation + white
    {"name": "rotate_white", "template": "\n\\rotatebox{{180}}{{\\textcolor{{white}}{{{payload}}}}}\n"},
    {"name": "reflect_white", "template": "\n\\reflectbox{{\\textcolor{{white}}{{{payload}}}}}\n"},

    # Opacity/Transparency (novel - not in known sanitizer)
    {"name": "transparent_zero", "template": "\n{{\\transparent{{0}}{payload}}}\n"},
    {"name": "texttransparent", "template": "\n\\texttransparent{{0}}{{{payload}}}\n"},

    # Multi-layer wrapping (harder to detect)
    {"name": "layered_hide", "template": "\n\\makebox[0pt]{{\\scalebox{{0.001}}{{\\textcolor{{white}}{{{payload}}}}}}}\n"},
]

# Injection points (where to insert payload in the LaTeX document)
INJECTION_POINTS = [
    (r"\\end\{abstract\}", "after", "right after abstract"),
    (r"\\section\{Introduction", "before", "right before Introduction"),
    (r"\\section\{Method", "after", "right after Method heading"),
    (r"\\section\{Results", "after", "right after Results heading"),
    (r"\\section\{Experimental", "after", "right after Experimental Setup"),
    (r"\\section\{Conclusion", "before", "right before Conclusions"),
    (r"\\end\{document\}", "before", "at end of document"),
    (r"\\begin\{figure\}", "before", "before first figure"),
    (r"\\begin\{table\}", "before", "before first table"),
]


# ---- Novel technique discovery ----

TECHNIQUE_EXPLORATION_POOL = [
    # Color space variants (not caught by {white} or [HTML]{FFFFFF})
    "\n\\textcolor[cmyk]{{0,0,0,0}}{{{payload}}}\n",
    "\n\\textcolor[gray]{{1}}{{{payload}}}\n",
    "\n\\textcolor[rgb]{{1,1,1}}{{{payload}}}\n",
    "\n{{\\color[HTML]{{FFFFFF}}{payload}}}\n",

    # ocg (optional content groups) — hides text in PDF layers
    "\n\\begin{{ocg}}{{hidden}}{{hide}}{{0}}\n{payload}\n\\end{{ocg}}\n",

    # accsupp (accessibility text) — may survive extraction differently
    "\n\\BeginAccSupp{{ActualText={payload}}}\n\\EndAccSupp{{}}\n",

    # marginpar trick (puts text in margin then pushes it off)
    "\n\\marginpar{{\\hspace{{20cm}}\\textcolor{{white}}{{{payload}}}}}\n",

    # pdfcomment hidden note
    "\n\\pdfcomment[icon=Note,color=white,opacity=0]{{{payload}}}\n",

    # watermark trick
    "\n\\SetWatermarkText{{\\textcolor{{white}}{{{payload}}}}}\n",

    # Combined vspace + kern + color
    "\n\\vspace*{{-30cm}}\\kern-50cm\\textcolor{{white}}{{{payload}}}\\kern50cm\\vspace*{{30cm}}\n",
]


class HidingSelector:
    """Selects LaTeX hiding techniques for wrapping injection payloads.

    Maintains a pool of techniques (both known and explored), and can
    systematically discover new variants by combining LaTeX commands
    in novel ways.
    """

    def __init__(self, exploration_rate: float = 0.3):
        self.known_techniques = list(KNOWN_HIDING_TECHNIQUES)
        self.exploration_pool = list(TECHNIQUE_EXPLORATION_POOL)
        self.exploration_rate = exploration_rate
        self.success_counts: Dict[str, int] = {}  # technique -> successful bypasses

    def get_technique(self, name: str) -> Optional[Dict]:
        """Get a technique by name."""
        for t in self.known_techniques:
            if t["name"] == name:
                return t
        return None

    def sample_technique(self, prefer_successful: bool = True) -> Dict:
        """Sample a hiding technique.

        With exploration_rate probability, samples from the exploration
        pool (novel techniques). Otherwise, samples from known techniques,
        weighted by past success counts.

        Args:
            prefer_successful: If True, weight known techniques by success count.

        Returns:
            Technique dict with 'name' and 'template' keys.
        """
        # Exploration: try novel techniques
        if random.random() < self.exploration_rate and self.exploration_pool:
            template = random.choice(self.exploration_pool)
            return {"name": "exploration", "template": template}

        # Exploitation: use known techniques
        if prefer_successful and self.success_counts:
            # Weight by log(success + 1) to avoid dominance by one technique
            weights = [
                max(0.1, self.success_counts.get(t["name"], 1))
                for t in self.known_techniques
            ]
            total = sum(weights)
            probs = [w / total for w in weights]
            return random.choices(self.known_techniques, weights=probs, k=1)[0]
        else:
            return random.choice(self.known_techniques)

    def sample_injection_point(self) -> Tuple[str, str, str]:
        """Sample an injection point.

        Returns:
            (latex_pattern, position_type, description)
        """
        return random.choice(INJECTION_POINTS)

    def wrap_payload(self, payload: str, technique: Optional[Dict] = None) -> str:
        """Wrap a payload in a LaTeX hiding technique.

        Args:
            payload: The attack payload text.
            technique: Technique dict. If None, samples one.

        Returns:
            LaTeX snippet with the payload wrapped in hiding commands.
        """
        if technique is None:
            technique = self.sample_technique()

        template = technique["template"]
        # Sanitize payload for LaTeX special chars
        safe_payload = payload.replace("\\", "\\\\").replace("#", "\\#")
        return template.replace("{payload}", safe_payload)

    def inject_into_latex(
        self,
        latex_content: str,
        payload: str,
        technique: Optional[Dict] = None,
        injection_point: Optional[Tuple[str, str, str]] = None,
    ) -> str:
        """Inject a wrapped payload into a LaTeX document.

        Args:
            latex_content: Original LaTeX source.
            payload: Attack payload text.
            technique: Hiding technique. Sampled if None.
            injection_point: Where to inject. Sampled if None.

        Returns:
            Modified LaTeX with injected payload.
        """
        wrapped = self.wrap_payload(payload, technique)

        # Collect candidate injection points (specified one first, then shuffled rest)
        candidates = []
        if injection_point is not None:
            candidates.append(injection_point)
        shuffled = list(INJECTION_POINTS)
        random.shuffle(shuffled)
        for pt in shuffled:
            if pt not in candidates:
                candidates.append(pt)

        # Try each injection point until one matches
        for pattern, position, _ in candidates:
            if not re.search(pattern, latex_content, flags=re.DOTALL):
                continue

            if position == "after":
                modified = re.sub(
                    pattern,
                    lambda m: m.group(0) + wrapped,
                    latex_content,
                    count=1,
                    flags=re.DOTALL,
                )
            else:  # "before"
                modified = re.sub(
                    pattern,
                    lambda m: wrapped + m.group(0),
                    latex_content,
                    count=1,
                    flags=re.DOTALL,
                )
            return modified

        # Fallback: inject before \end{document}
        if r"\end{document}" in latex_content:
            modified = latex_content.replace(
                r"\end{document}", wrapped + r"\end{document}", 1
            )
            return modified

        # Ultimate fallback: append to end
        return latex_content + wrapped

    def register_success(self, technique_name: str, bypassed: bool):
        """Update technique success stats."""
        if technique_name not in self.success_counts:
            self.success_counts[technique_name] = 0
        if bypassed:
            self.success_counts[technique_name] += 1

    def get_stats(self) -> Dict[str, int]:
        """Get technique success statistics."""
        return dict(self.success_counts)

    def set_exploration_rate(self, rate: float):
        """Dynamically adjust exploration rate."""
        self.exploration_rate = max(0.05, min(1.0, rate))
