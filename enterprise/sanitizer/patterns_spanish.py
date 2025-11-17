#!/usr/bin/env python3
"""Spanish language patterns loaded from external bundle."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List

_BUNDLE_PATH = Path(__file__).with_name("pattern_bundles") / "es_core.json"

with _BUNDLE_PATH.open("r", encoding="utf-8") as fh:
    _DATA: Dict[str, Any] = json.load(fh)

SPANISH_KEYWORDS: List[str] = list(_DATA.get("keywords", []))
SPANISH_STRUCTURE: List[str] = list(_DATA.get("structure", []))
SPANISH_ENTROPY_TRIGGERS: List[str] = list(_DATA.get("entropy_triggers", []))
SPANISH_METADATA: Dict[str, Any] = dict(_DATA.get("metadata", {}))
SPANISH_ALL_PATTERNS: List[str] = SPANISH_KEYWORDS + SPANISH_STRUCTURE

__all__ = [
    "SPANISH_KEYWORDS",
    "SPANISH_STRUCTURE",
    "SPANISH_ENTROPY_TRIGGERS",
    "SPANISH_METADATA",
    "SPANISH_ALL_PATTERNS",
]
