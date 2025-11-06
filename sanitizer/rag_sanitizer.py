#!/usr/bin/env python3
"""Community edition query sanitizer.

The goal of the open build is to stay lightweight and dependency-free.
We retain the pattern gate to detect risky scaffolds and apply a simple
rewrite that strips those scaffolds before handing the query to the
embedder. Advanced model-assisted rewriting, PHI masking, telemetry, and
rate-limiting live in the enterprise repository.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Tuple
import re

from sanitizer.jailbreak.prr_gate import PRRGate, DEFAULT_KEYWORDS, DEFAULT_STRUCTURE


@dataclass
class SanitizerConfig:
    """Configuration for the minimal sanitizer."""

    keyword_patterns: Iterable[str] = field(default_factory=lambda: DEFAULT_KEYWORDS)
    structure_patterns: Iterable[str] = field(default_factory=lambda: DEFAULT_STRUCTURE)
    # Number of pattern families that must trigger before we treat a query as risky
    min_signals: int = 1
    # Regexes removed from the risky query when sanitising
    removal_patterns: Iterable[str] = field(default_factory=lambda: DEFAULT_KEYWORDS)


class QuerySanitizer:
    """Simple regex-based sanitizer used in the open-source edition."""

    def __init__(self, cfg: SanitizerConfig | None = None) -> None:
        self.cfg = cfg or SanitizerConfig()
        self.prr = PRRGate(
            keyword_patterns=self.cfg.keyword_patterns,
            structure_patterns=self.cfg.structure_patterns,
            min_signals=self.cfg.min_signals,
        )
        self._removal_regex: List[re.Pattern[str]] = [
            re.compile(pattern, re.IGNORECASE) for pattern in self.cfg.removal_patterns
        ]

    def sanitize_query(self, query: str) -> Tuple[str, Dict[str, object]]:
        """Return a sanitised query and metadata for downstream systems."""
        evaluation = self.prr.evaluate(query)
        if evaluation.risky:
            sanitized = self._strip_scaffolds(query)
        else:
            sanitized = query

        metadata: Dict[str, object] = {
            "risky": evaluation.risky,
            "keyword_hits": evaluation.keyword_hits,
            "structure_hits": evaluation.structure_hits,
            "score": evaluation.score,
            "sanitized": sanitized != query,
        }
        return sanitized, metadata

    def _strip_scaffolds(self, text: str) -> str:
        output = text
        for rx in self._removal_regex:
            output = rx.sub("", output)
        # collapse whitespace
        output = re.sub(r"\s+", " ", output)
        return output.strip()


__all__ = ["QuerySanitizer", "SanitizerConfig"]
