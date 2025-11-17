#!/usr/bin/env python3
"""Lightweight pattern gate used by the open-source RagWall build.

Re-implements the original multi-language + healthcare-aware PRR gate
from the patent draft. The module keeps regex-only execution but now
supports:

* Language-specific pattern bundles with auto-detection
* Healthcare-mode bundles that add HIPAA / PHI heuristics
* Compliance receipts with optional Ed25519 signatures
* Metadata describing which signal families fired
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List
import json
import re

from sanitizer.utils.receipts import ComplianceReceiptBuilder, ReceiptConfig
from sanitizer.utils.text_normalization import (
    compute_obfuscation_ratio,
    normalize_text_for_detection,
)

_BUNDLE_DIR = Path(__file__).with_name("pattern_bundles")
_SUPPORTED_LANGS = ("en", "es", "fr", "de", "pt")
_HEALTHCARE_RULES: List[tuple[str, re.Pattern[str]]] = [
    ("HIPAA-violation", re.compile(r"hipaa|45\s*cfr|phi", re.I)),
    ("PHI-disclosure", re.compile(r"ssn|medical\s+record|patient|dea", re.I)),
    ("unauthorized-access", re.compile(r"doctor|nurse|admin|credential|password", re.I)),
    ("data-exfiltration", re.compile(r"export|download|dump|exfiltrate", re.I)),
]


def _load_bundle(name: str) -> Dict[str, Any]:
    bundle_path = _BUNDLE_DIR / f"{name}.json"
    if not bundle_path.exists():
        raise FileNotFoundError(f"Pattern bundle not found: {bundle_path}")
    with bundle_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


_en_core = _load_bundle("en_core")
DEFAULT_KEYWORDS: List[str] = list(_en_core.get("keywords", []))
DEFAULT_STRUCTURE: List[str] = list(_en_core.get("structure", []))

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


def detect_language(text: str) -> str:
    """Naive heuristic language detector (mirrors enterprise edition)."""
    lowered = text.lower()
    tokens = Counter(_TOKEN_RE.findall(lowered))

    def score(lang: str) -> int:
        char_hits = sum(lowered.count(ch) for ch in _LANGUAGE_CHARS.get(lang, ""))
        word_hits = sum(tokens.get(word, 0) for word in _LANGUAGE_WORDS.get(lang, []))
        return char_hits + word_hits

    scores = {lang: score(lang) for lang in _LANGUAGE_WORDS}
    lang, value = max(scores.items(), key=lambda kv: kv[1])
    return lang if value > 1 else "en"


def load_language_patterns(language: str, healthcare_mode: bool = False) -> Dict[str, List[str]]:
    base_lang = language if language in _SUPPORTED_LANGS else "en"
    try:
        core = _load_bundle(f"{base_lang}_core")
    except FileNotFoundError:
        core = _load_bundle("en_core")

    keywords = list(core.get("keywords", []))
    structure = list(core.get("structure", []))

    if healthcare_mode:
        for candidate in (f"{base_lang}_healthcare", "en_healthcare"):
            try:
                bundle = _load_bundle(candidate)
                keywords.extend(bundle.get("keywords", []))
                structure.extend(bundle.get("structure", []))
                break
            except FileNotFoundError:
                continue

    return {"keywords": keywords, "structure": structure}


def _healthcare_families(hits: Iterable[str], text: str = "") -> List[str]:
    families: List[str] = []
    for label, pattern in _HEALTHCARE_RULES:
        if any(pattern.search(hit) for hit in hits) or (text and pattern.search(text)):
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
    transformer_score: float | None = None
    obfuscation_detected: bool = False
    obfuscation_ratio: float = 0.0


class PRRGate:
    """Regex-first gate with multi-language + healthcare awareness."""

    def __init__(
        self,
        keyword_patterns: Iterable[str] | None = None,
        structure_patterns: Iterable[str] | None = None,
        *,
        keyword_threshold: int = 1,
        structure_threshold: int = 1,
        min_signals: int = 1,
        language: str = "en",
        healthcare_mode: bool = False,
        auto_detect_language: bool = False,
        receipt_config: ReceiptConfig | None = None,
        domain: str | None = None,
        transformer_fallback: bool = False,
        transformer_model_name: str = "ProtectAI/deberta-v3-base-prompt-injection-v2",
        transformer_threshold: float = 0.5,
        transformer_device: str | None = None,
        transformer_domain_tokens: Dict[str, str] | None = None,
        transformer_domain_thresholds: Dict[str, float] | None = None,
        obfuscation_threshold: float | None = 0.25,
        obfuscation_counts_as_signal: bool = True,
    ) -> None:
        self.keyword_threshold = max(0, int(keyword_threshold))
        self.structure_threshold = max(0, int(structure_threshold))
        self.min_signals = max(1, int(min_signals))
        self.language = language
        self.healthcare_mode = healthcare_mode
        self.auto_detect_language = auto_detect_language
        self._receipt_builder = ComplianceReceiptBuilder(receipt_config)
        self.domain = domain
        self.transformer_fallback = transformer_fallback
        self.transformer_model_name = transformer_model_name
        self.transformer_threshold = float(transformer_threshold)
        self.transformer_device = transformer_device
        self.transformer_domain_tokens = transformer_domain_tokens
        self.transformer_domain_thresholds = transformer_domain_thresholds
        self._transformer_classifier = None
        self.obfuscation_threshold = obfuscation_threshold if (obfuscation_threshold or 0) > 0 else None
        self.obfuscation_counts_as_signal = obfuscation_counts_as_signal

        if keyword_patterns is not None or structure_patterns is not None:
            keywords = list(keyword_patterns) if keyword_patterns is not None else []
            structure = list(structure_patterns) if structure_patterns is not None else []
        else:
            patterns = load_language_patterns(language, healthcare_mode)
            keywords = patterns["keywords"]
            structure = patterns["structure"]

        self._keyword_regex: List[re.Pattern[str]] = [re.compile(pat, re.IGNORECASE) for pat in keywords]
        self._structure_regex: List[re.Pattern[str]] = [re.compile(pat, re.IGNORECASE) for pat in structure]

    def _reload_language(self, language: str) -> None:
        patterns = load_language_patterns(language, self.healthcare_mode)
        self._keyword_regex = [re.compile(pat, re.IGNORECASE) for pat in patterns["keywords"]]
        self._structure_regex = [re.compile(pat, re.IGNORECASE) for pat in patterns["structure"]]
        self.language = language

    def _run(self, text: str, patterns: List[re.Pattern[str]]) -> List[str]:
        hits: List[str] = []
        for rx in patterns:
            try:
                if rx.search(text):
                    hits.append(rx.pattern)
            except re.error:
                continue
        return hits

    def evaluate(self, text: str) -> PRRResult:
        if self.auto_detect_language:
            detected = detect_language(text)
            if detected != self.language:
                self._reload_language(detected)

        # Normalize text to handle obfuscation attacks (leetspeak, unicode homoglyphs, etc.)
        normalized_text = normalize_text_for_detection(text, aggressive=True)
        obfuscation_ratio = compute_obfuscation_ratio(text, normalized_text)
        obfuscation_signal = bool(
            self.obfuscation_threshold is not None and obfuscation_ratio >= self.obfuscation_threshold
        )

        keyword_hits = self._run(normalized_text, self._keyword_regex)
        structure_hits = self._run(normalized_text, self._structure_regex)

        keyword_signal = bool(self.keyword_threshold and len(keyword_hits) >= self.keyword_threshold)
        structure_signal = bool(self.structure_threshold and len(structure_hits) >= self.structure_threshold)
        families: List[str] = []
        if keyword_signal:
            families.append("keyword")
        if structure_signal:
            families.append("structure")
        if obfuscation_signal:
            families.append("obfuscation")
        risky = (int(keyword_signal) + int(structure_signal)) >= self.min_signals
        if obfuscation_signal and self.obfuscation_counts_as_signal:
            risky = True
        transformer_score: float | None = None
        transformer_triggered = False
        if not risky and self.transformer_fallback:
            classifier = self._get_transformer_classifier()
            transformer_triggered, transformer_score = classifier.classify(
                text, self.transformer_threshold, domain=self.domain
            )
            if transformer_triggered:
                risky = True
                families.append("transformer")

        healthcare_fam = (
            _healthcare_families(keyword_hits + structure_hits, normalized_text)
            if self.healthcare_mode
            else []
        )
        score = float(len(keyword_hits) + len(structure_hits))
        receipt = self._receipt_builder.build(
            event_type="prr_gate.community",
            query=text,
            sanitized=None,
            risky=risky,
            families=families,
            language=self.language,
            healthcare_mode=self.healthcare_mode,
            extra={
                "healthcare_families": healthcare_fam,
                "hits": len(keyword_hits) + len(structure_hits),
                "transformer_score": transformer_score,
                "transformer_triggered": transformer_triggered,
                "obfuscation_detected": obfuscation_signal,
                "obfuscation_ratio": obfuscation_ratio,
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
            transformer_score=transformer_score,
            obfuscation_detected=obfuscation_signal,
            obfuscation_ratio=obfuscation_ratio,
        )

    def _get_transformer_classifier(self):
        if not self.transformer_fallback:
            return None
        if self._transformer_classifier is None:
            from sanitizer.ml.transformer_fallback import TransformerPromptInjectionClassifier

            self._transformer_classifier = TransformerPromptInjectionClassifier(
                model_name=self.transformer_model_name,
                device=self.transformer_device,
                domain_tokens=self.transformer_domain_tokens,
                domain_thresholds=self.transformer_domain_thresholds,
            )
        return self._transformer_classifier


__all__ = [
    "PRRGate",
    "PRRResult",
    "DEFAULT_KEYWORDS",
    "DEFAULT_STRUCTURE",
    "ReceiptConfig",
]
