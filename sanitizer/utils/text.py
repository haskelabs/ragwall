"""Text helpers shared by the sanitizer modules."""
from __future__ import annotations

import re
import unicodedata

ZERO_WIDTH_CHARS = "\u200b\u200c\u200d\u2060\ufeff"
_WHITESPACE_RE = re.compile(r"\s+")
_TRANSLATION_TABLE = {ord(ch): "" for ch in ZERO_WIDTH_CHARS}


def strip_zero_width(value: str) -> str:
    """Remove zero-width characters to stabilize canonicalization."""
    return value.translate(_TRANSLATION_TABLE)


def canonicalize_query(value: str) -> str:
    """Return a normalized form for equality/embedding checks."""
    if not value:
        return ""
    text = unicodedata.normalize("NFKC", value)
    text = strip_zero_width(text)
    text = text.lower()
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip()


__all__ = ["canonicalize_query", "strip_zero_width", "ZERO_WIDTH_CHARS"]
