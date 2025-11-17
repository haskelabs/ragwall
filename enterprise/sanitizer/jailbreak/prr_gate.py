#!/usr/bin/env python3
"""Lightweight pattern gate used by the open-source RagWall build.

This module keeps the English core jailbreak patterns and exposes a small
helper class for scoring queries. The enterprise edition layers in
additional languages, cosine similarity signals, and healthcare bundles –
all of that logic now lives in the private codebase.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple
import json
import math
import re
import sys

try:
    import torch  # type: ignore
except Exception:
    torch = None  # type: ignore

CURRENT_DIR = Path(__file__).resolve()
REPO_ROOT = CURRENT_DIR.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sanitizer.utils.receipts import ComplianceReceiptBuilder, ReceiptConfig

_BUNDLE_DIR = Path(__file__).with_name("pattern_bundles")
_HEALTHCARE_RULES: List[tuple[str, re.Pattern[str]]] = [
    ("HIPAA-violation", re.compile(r"hipaa|45\s*cfr|phi", re.I)),
    ("PHI-disclosure", re.compile(r"ssn|medical\s+record|patient|dea", re.I)),
    ("unauthorized-access", re.compile(r"doctor|nurse|admin|credential|password", re.I)),
    ("data-exfiltration", re.compile(r"export|download|dump|exfiltrate", re.I)),
]


def sigmoid(x: float, k: float = 8.0) -> float:
    """Sigmoid function for smooth probability mapping."""
    return float(1.0 / (1.0 + math.exp(-k * x)))


def normalize01(x: float, lo: float, hi: float) -> float:
    """Normalize value to [0, 1] range."""
    if hi <= lo:
        return 0.0
    return max(0.0, min(1.0, (x - lo) / (hi - lo)))


def _load_bundle(name: str) -> Dict[str, Any]:
    bundle_path = _BUNDLE_DIR / f"{name}.json"
    if not bundle_path.exists():
        raise FileNotFoundError(f"Pattern bundle not found: {bundle_path}")
    with bundle_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


_en_core = _load_bundle("en_core")
DEFAULT_KEYWORDS: List[str] = list(_en_core.get("keywords", []))
DEFAULT_STRUCTURE: List[str] = list(_en_core.get("structure", []))


def _bundle_lists() -> List[str]:
    """List available pattern bundles."""
    if not _BUNDLE_DIR.exists():
        return []
    return [p.stem for p in _BUNDLE_DIR.glob("*.json")]


def detect_language(text: str) -> str:
    """Detect language of input text using simple heuristics."""
    lowered = text.lower()
    tokens = Counter(_TOKEN_RE.findall(lowered))

    def score(lang: str) -> int:
        char_hits = sum(lowered.count(ch) for ch in _LANGUAGE_CHARS.get(lang, ""))
        word_hits = sum(tokens.get(word, 0) for word in _LANGUAGE_WORDS.get(lang, []))
        return char_hits + word_hits

    scores = {lang: score(lang) for lang in _LANGUAGE_WORDS}
    lang, value = max(scores.items(), key=lambda kv: kv[1])
    return lang if value > 1 else 'en'


def load_language_patterns(language: str, healthcare_mode: bool = False) -> Dict[str, List[str]]:
    """Load patterns for a specific language.

    Args:
        language: Language code (e.g., 'en', 'es', 'fr')
        healthcare_mode: Whether to load healthcare-specific patterns

    Returns:
        Dictionary with 'keywords' and 'structure' pattern lists
    """
    # Load core patterns
    core_bundle = _load_bundle(f"{language}_core")
    keywords = list(core_bundle.get("keywords", []))
    structure = list(core_bundle.get("structure", []))

    # Add healthcare patterns if requested
    if healthcare_mode:
        try:
            healthcare_bundle = _load_bundle(f"{language}_healthcare")
            keywords.extend(healthcare_bundle.get("keywords", []))
            structure.extend(healthcare_bundle.get("structure", []))
        except FileNotFoundError:
            # Healthcare bundle not available for this language
            pass

    return {"keywords": keywords, "structure": structure}


def _healthcare_families(hits: Iterable[str]) -> List[str]:
    families: List[str] = []
    for label, pattern in _HEALTHCARE_RULES:
        if any(pattern.search(hit) for hit in hits):
            families.append(label)
    return sorted(set(families))


@dataclass
class PRRResult:
    risky: bool
    keyword_hits: List[str]
    structure_hits: List[str]
    score: float
    families_hit: List[str] = field(default_factory=list)
    detected_language: str = "en"
    healthcare_families: List[str] = field(default_factory=list)
    compliance_receipt: Dict[str, Any] | None = None


@dataclass
class PRRScoreResult:
    """Extended result for score() method with ML-enhanced scoring."""
    risky: bool
    keyword_hits: List[str]
    structure_hits: List[str]
    score: float
    families_hit: List[str]
    details: Dict[str, Any]
    p_keyword: float = 0.0
    p_cosine: float = 0.0
    p_entropy: float = 0.0
    p_structure: float = 0.0
    p_missing_self: float = 0.0


class PRRGate:
    """Simplified pattern gate for the community edition.

    The open build keeps things deterministic and dependency-free by using
    plain regular expressions. Each positive keyword or structural hit adds
    to a simple score. A query is marked risky when a minimum number of
    signals fire (defaults to one keyword or structure match).

    Enterprise edition adds language detection and healthcare mode support.
    """

    def __init__(
        self,
        keyword_patterns: Iterable[str] | None = None,
        structure_patterns: Iterable[str] | None = None,
        *,
        keyword_threshold: int = 1,
        structure_threshold: int = 1,
        min_signals: int = 1,
        language: str = 'en',
        healthcare_mode: bool = False,
        auto_detect_language: bool = False,
        receipt_config: ReceiptConfig | None = None,
    ) -> None:
        self.keyword_threshold = max(0, int(keyword_threshold))
        self.structure_threshold = max(0, int(structure_threshold))
        self.min_signals = max(1, int(min_signals))
        self.language = language
        self.healthcare_mode = healthcare_mode
        self.auto_detect_language = auto_detect_language
        self._receipt_builder = ComplianceReceiptBuilder(receipt_config)

        # If patterns are explicitly provided, use them
        if keyword_patterns is not None or structure_patterns is not None:
            keywords = list(keyword_patterns) if keyword_patterns is not None else DEFAULT_KEYWORDS
            structure = list(structure_patterns) if structure_patterns is not None else DEFAULT_STRUCTURE
        else:
            # Load patterns based on language and healthcare mode
            patterns = load_language_patterns(language, healthcare_mode)
            keywords = patterns['keywords']
            structure = patterns['structure']

        self._keyword_regex: List[re.Pattern[str]] = [re.compile(pat, re.IGNORECASE) for pat in keywords]
        self._structure_regex: List[re.Pattern[str]] = [re.compile(pat, re.IGNORECASE) for pat in structure]
        self.device = None  # Set device for cosine similarity
        self.cosine_tau = 0.25  # Threshold for cosine similarity

    def _run(self, text: str, patterns: List[re.Pattern[str]]) -> List[str]:
        hits: List[str] = []
        for rx in patterns:
            try:
                if rx.search(text):
                    hits.append(rx.pattern)
            except re.error:
                # If a regex fails to compile in downstream custom configs we
                # fail closed by ignoring the specific pattern.
                continue
        return hits

    def evaluate(self, text: str) -> PRRResult:
        # Auto-detect language if enabled
        if self.auto_detect_language:
            detected_lang = detect_language(text)
            if detected_lang != self.language:
                # Reload patterns for detected language
                patterns = load_language_patterns(detected_lang, self.healthcare_mode)
                self._keyword_regex = [re.compile(pat, re.IGNORECASE) for pat in patterns['keywords']]
                self._structure_regex = [re.compile(pat, re.IGNORECASE) for pat in patterns['structure']]
                self.language = detected_lang

        keyword_hits = self._run(text, self._keyword_regex)
        structure_hits = self._run(text, self._structure_regex)

        keyword_signal = len(keyword_hits) >= self.keyword_threshold if self.keyword_threshold else False
        structure_signal = len(structure_hits) >= self.structure_threshold if self.structure_threshold else False
        families: List[str] = []
        if keyword_signal:
            families.append('keyword')
        if structure_signal:
            families.append('structure')
        num_signals = int(keyword_signal) + int(structure_signal)
        risky = num_signals >= self.min_signals

        # Score is a simple aggregate used for telemetry / debugging.
        score = float(len(keyword_hits) + len(structure_hits))
        healthcare_fam = _healthcare_families(keyword_hits + structure_hits) if self.healthcare_mode else []
        receipt = self._receipt_builder.build(
            event_type="prr_gate.enterprise",
            query=text,
            sanitized=None,
            risky=risky,
            families=families,
            language=self.language,
            healthcare_mode=self.healthcare_mode,
            extra={
                "healthcare_families": healthcare_fam,
                "hits": len(keyword_hits) + len(structure_hits),
            },
        )
        return PRRResult(
            risky=risky,
            keyword_hits=keyword_hits,
            structure_hits=structure_hits,
            score=score,
            families_hit=families,
            detected_language=self.language,
            healthcare_families=healthcare_fam,
            compliance_receipt=receipt,
        )

    def _keyword_prob(self, text: str) -> Tuple[float, List[str]]:
        """Calculate keyword detection probability."""
        hits = [p.pattern for p in self._keyword_regex if p.search(text)]
        # Map count to probability: 1 - prod(1-0.4)^m
        m = len(hits)
        p = 1.0 - (0.6 ** m)
        return float(p), hits

    def _structure_prob(self, text: str) -> Tuple[float, List[str]]:
        """Calculate structure detection probability."""
        hits = [p.pattern for p in self._structure_regex if p.search(text)]
        # Map count to probability: 1 - prod(1-0.3)^m
        m = len(hits)
        p = 1.0 - (0.7 ** m)
        return float(p), hits

    def _cosine_prob(self, pooled: 'torch.Tensor', per_pattern: Dict[str, 'torch.Tensor']) -> Tuple[float, Dict[str, float]]:
        """Calculate cosine similarity probability against pattern vectors."""
        if torch is None or pooled is None or (hasattr(pooled, 'numel') and pooled.numel() == 0) or not per_pattern:
            return 0.0, {}

        x = pooled / (pooled.norm(p=2) + 1e-8)
        scores: Dict[str, float] = {}

        for name, vec in per_pattern.items():
            if not isinstance(vec, torch.Tensor):
                continue
            v = vec.to(self.device if self.device is not None else pooled.device)
            v = v / (v.norm(p=2) + 1e-8)
            c = float(torch.dot(x, v).item())
            scores[name] = c

        # Take max over patterns and map to prob via normalized ramp around tau
        mx = max(scores.values()) if scores else 0.0
        p = normalize01(mx, self.cosine_tau, self.cosine_tau + 0.25)
        return float(p), scores

    def _missing_self_prob(self, text: str, kw_hits: List[str], struct_hits: List[str]) -> float:
        """Detect queries that override instructions but lack genuine task intent."""
        t = text.strip()
        override = (len(kw_hits) > 0) or (len(struct_hits) > 0)
        no_question = ('?' not in t)
        stripped = (len(t) < 24) or bool(re.search(r"^(ignore|disregard|act as|from now on)", t, re.I))
        roleplay_no_task = bool(re.search(r"role.?play", t, re.I)) and not bool(re.search(r"\b(explain|write|how|why|list|show)\b", t, re.I))

        if override and (no_question or stripped or roleplay_no_task):
            return 0.7
        return 0.0

    def score(self, text: str, pooled_state=None, meta: Dict[str, Any] = None, boundary_entropy_conf: float = 0.0) -> PRRScoreResult:
        """ML-enhanced scoring with multiple signal families.

        Args:
            text: Query text to evaluate
            pooled_state: Optional torch tensor for cosine similarity
            meta: Additional metadata (optional)
            boundary_entropy_conf: Entropy confidence score [0, 1]

        Returns:
            PRRScoreResult with multi-signal probabilities and risk assessment
        """
        # Get regex-based probabilities
        p_kw, kw_hits = self._keyword_prob(text)
        p_st, st_hits = self._structure_prob(text)

        # Get ML-based probabilities if available
        per_pattern = (meta or {}).get('per_pattern_vectors', {})
        p_cs, cs_map = self._cosine_prob(pooled_state, per_pattern)

        # Entropy-based probability
        p_en = float(max(0.0, min(1.0, boundary_entropy_conf)))

        # Missing self-reference detection
        p_ms = self._missing_self_prob(text, kw_hits, st_hits)

        # Determine which families fired (threshold = 0.3)
        TH = 0.3
        families_hit = []
        if p_kw >= TH:
            families_hit.append('keyword')
        if p_cs >= TH:
            families_hit.append('cosine')
        if p_st >= TH:
            families_hit.append('structure')
        if p_en >= 0.6:
            families_hit.append('entropy')
        if p_ms >= 0.5:
            families_hit.append('missing_self')

        # Determine if risky (quorum: need 2+ signals OR very high single signal)
        risky = len(families_hit) >= 2 or max(p_kw, p_cs, p_st, p_en, p_ms) >= 0.75

        # Aggregate score
        score = p_kw + p_cs + p_st + p_en + p_ms

        details = {
            'kw_hits': kw_hits,
            'struct_hits': st_hits,
            'cosine': cs_map,
            'detected_language': self.language if self.auto_detect_language else 'en',
            'healthcare_mode': self.healthcare_mode,
        }

        return PRRScoreResult(
            risky=risky,
            keyword_hits=kw_hits,
            structure_hits=st_hits,
            score=score,
            families_hit=families_hit,
            details=details,
            p_keyword=p_kw,
            p_cosine=p_cs,
            p_entropy=p_en,
            p_structure=p_st,
            p_missing_self=p_ms
        )


__all__ = [
    "PRRGate",
    "PRRResult",
    "PRRScoreResult",
    "DEFAULT_KEYWORDS",
    "DEFAULT_STRUCTURE",
    "load_language_patterns",
    "detect_language",
    "_load_bundle",
    "_bundle_lists",
    "ReceiptConfig",
]
_LANGUAGE_CHARS: Dict[str, str] = {
    "es": "ñáéíóúü¿¡",
    "fr": "àâçéèêëîïôûù",
    "de": "äöüß",
    "pt": "ãõçáéíóú",
}

_LANGUAGE_WORDS: Dict[str, List[str]] = {
    "es": [
        "el",
        "la",
        "los",
        "las",
        "de",
        "que",
        "para",
        "con",
        "por",
        "del",
        "y",
        "sin",
        "favor",
        "todas",
        "todos",
        "instrucciones",
    ],
    "fr": ["le", "la", "les", "des", "une", "et", "pour", "dans"],
    "de": ["der", "die", "das", "und", "ist", "für", "mit"],
    "pt": ["o", "a", "os", "as", "que", "com", "uma", "para", "sem"],
}

_TOKEN_RE = re.compile(r"[\wñáéíóúüçàâçéèêëîïôûùãõäöüß]+", re.UNICODE)
