#!/usr/bin/env python3
"""Lightweight pattern gate used by the open-source RagWall build.

This module keeps the English core jailbreak patterns and exposes a small
helper class for scoring queries. The enterprise edition layers in
additional languages, cosine similarity signals, and healthcare bundles –
all of that logic now lives in the private codebase.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List
import json
import re

_BUNDLE_DIR = Path(__file__).with_name("pattern_bundles")


def _load_bundle(name: str) -> Dict[str, Any]:
    bundle_path = _BUNDLE_DIR / f"{name}.json"
    if not bundle_path.exists():
        raise FileNotFoundError(f"Pattern bundle not found: {bundle_path}")
    with bundle_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


_en_core = _load_bundle("en_core")
DEFAULT_KEYWORDS: List[str] = list(_en_core.get("keywords", []))
DEFAULT_STRUCTURE: List[str] = list(_en_core.get("structure", []))


@dataclass
class PRRResult:
    risky: bool
    keyword_hits: List[str]
    structure_hits: List[str]
    score: float


class PRRGate:
    """Simplified pattern gate for the community edition.

    The open build keeps things deterministic and dependency-free by using
    plain regular expressions. Each positive keyword or structural hit adds
    to a simple score. A query is marked risky when a minimum number of
    signals fire (defaults to one keyword or structure match).
    """

    def __init__(
        self,
        keyword_patterns: Iterable[str] | None = None,
        structure_patterns: Iterable[str] | None = None,
        *,
        keyword_threshold: int = 1,
        structure_threshold: int = 1,
        min_signals: int = 1,
    ) -> None:
        self.keyword_threshold = max(0, int(keyword_threshold))
        self.structure_threshold = max(0, int(structure_threshold))
        self.min_signals = max(1, int(min_signals))

        keywords = list(keyword_patterns) if keyword_patterns is not None else DEFAULT_KEYWORDS
        structure = list(structure_patterns) if structure_patterns is not None else DEFAULT_STRUCTURE

        self._keyword_regex: List[re.Pattern[str]] = [re.compile(pat, re.IGNORECASE) for pat in keywords]
        self._structure_regex: List[re.Pattern[str]] = [re.compile(pat, re.IGNORECASE) for pat in structure]

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
        keyword_hits = self._run(text, self._keyword_regex)
        structure_hits = self._run(text, self._structure_regex)

        keyword_signal = len(keyword_hits) >= self.keyword_threshold if self.keyword_threshold else False
        structure_signal = len(structure_hits) >= self.structure_threshold if self.structure_threshold else False
        num_signals = int(keyword_signal) + int(structure_signal)
        risky = num_signals >= self.min_signals

        # Score is a simple aggregate used for telemetry / debugging.
        score = float(len(keyword_hits) + len(structure_hits))
        return PRRResult(risky=risky, keyword_hits=keyword_hits, structure_hits=structure_hits, score=score)


__all__ = [
    "PRRGate",
    "PRRResult",
    "DEFAULT_KEYWORDS",
    "DEFAULT_STRUCTURE",
]
