#!/usr/bin/env python3
"""
Text normalization utilities for handling obfuscation attacks.

Handles:
- Leetspeak/1337speak (4→a, 3→e, etc.)
- Unicode homoglyphs (α→a, etc.)
- Zero-width characters
- Mixed case obfuscation
"""
import re
import unicodedata
from itertools import zip_longest
from typing import Dict

# Leetspeak character mappings
LEETSPEAK_MAP: Dict[str, str] = {
    '0': 'o',
    '1': 'i',
    '3': 'e',
    '4': 'a',
    '5': 's',
    '7': 't',
    '8': 'b',
    '9': 'g',
    '@': 'a',
    '$': 's',
    '!': 'i',
    '|': 'i',
    '+': 't',
}

# Unicode homoglyphs (visually similar characters from different scripts)
HOMOGLYPH_MAP: Dict[str, str] = {
    # Greek
    'α': 'a', 'β': 'b', 'γ': 'g', 'δ': 'd', 'ε': 'e', 'ζ': 'z', 'η': 'h',
    'θ': 'th', 'ι': 'i', 'κ': 'k', 'λ': 'l', 'μ': 'm', 'ν': 'n', 'ξ': 'x',
    'ο': 'o', 'π': 'p', 'ρ': 'r', 'σ': 's', 'ς': 's', 'τ': 't', 'υ': 'u',
    'φ': 'ph', 'χ': 'ch', 'ψ': 'ps', 'ω': 'o',

    # Cyrillic (comprehensive lookalikes) - both lowercase and uppercase
    'а': 'a', 'А': 'A', 'в': 'b', 'В': 'B', 'с': 'c', 'С': 'C',
    'е': 'e', 'Е': 'E', 'н': 'h', 'Н': 'H', 'і': 'i', 'І': 'I',
    'о': 'o', 'О': 'O', 'р': 'p', 'Р': 'P', 'ѕ': 's', 'х': 'x',
    'Х': 'X', 'у': 'y', 'У': 'Y', 'к': 'k', 'К': 'K', 'м': 'm',
    'М': 'M', 'т': 't', 'Т': 'T', 'ѕ': 's', 'Ѕ': 'S', 'ј': 'j',
    'Ј': 'J', 'ғ': 'f', 'Ғ': 'F', 'д': 'd', 'Д': 'D', 'з': 'z',
    'З': 'Z', 'г': 'g', 'Г': 'G', 'ц': 'u', 'Ц': 'U', 'ь': 'b',
    'Ь': 'B', 'ъ': 'b', 'Ъ': 'B', 'л': 'l', 'Л': 'L',

    # Other common homoglyphs
    'ⅰ': 'i', 'ⅱ': 'ii', 'ⅲ': 'iii', 'ⅳ': 'iv', 'ⅴ': 'v',
    '∕': '/', '⁄': '/', '⧸': '/',
}

# Zero-width and invisible characters
ZERO_WIDTH_CHARS = [
    '\u200B',  # Zero width space
    '\u200C',  # Zero width non-joiner
    '\u200D',  # Zero width joiner
    '\uFEFF',  # Zero width no-break space
    '\u2060',  # Word joiner
    '\u180E',  # Mongolian vowel separator
]


def normalize_leetspeak(text: str) -> str:
    """
    Normalize leetspeak/1337speak to standard characters.

    Examples:
        "Byp4ss" → "Bypass"
        "H1PAA" → "HIPAA"
        "pr0vide" → "provide"
    """
    result = []
    for char in text:
        normalized = LEETSPEAK_MAP.get(char, char)
        result.append(normalized)
    return ''.join(result)


def normalize_homoglyphs(text: str) -> str:
    """
    Normalize unicode homoglyphs to ASCII equivalents.

    Examples:
        "Bypαss" → "Bypass" (Greek alpha)
        "Нello" → "Hello" (Cyrillic Н)
    """
    result = []
    for char in text:
        normalized = HOMOGLYPH_MAP.get(char, char)
        result.append(normalized)
    return ''.join(result)


def remove_zero_width(text: str) -> str:
    """Remove zero-width and invisible characters."""
    for char in ZERO_WIDTH_CHARS:
        text = text.replace(char, '')
    return text


def normalize_unicode(text: str) -> str:
    """Apply Unicode NFKC normalization."""
    return unicodedata.normalize('NFKC', text)


def normalize_text_for_detection(text: str, aggressive: bool = True) -> str:
    """
    Comprehensive text normalization for obfuscation-resistant detection.

    Args:
        text: Input text to normalize
        aggressive: If True, apply all normalizations including leetspeak.
                   If False, only apply unicode and homoglyph normalization.

    Returns:
        Normalized text suitable for pattern matching

    Examples:
        >>> normalize_text_for_detection("Byp4ss H1PAA")
        'bypass hipaa'

        >>> normalize_text_for_detection("Bypαss HIPAA")
        'bypass hipaa'

        >>> normalize_text_for_detection("B​y​p​a​s​s")  # with zero-width spaces
        'bypass'
    """
    # Step 1: Remove zero-width characters
    text = remove_zero_width(text)

    # Step 2: Unicode normalization (NFKC)
    text = normalize_unicode(text)

    # Step 3: Lowercase for case-insensitive matching
    text = text.lower()

    # Step 4: Normalize homoglyphs BEFORE leetspeak to avoid double replacements
    text = normalize_homoglyphs(text)

    # Step 5: Normalize leetspeak (if aggressive)
    if aggressive:
        text = normalize_leetspeak(text)

    # Step 6: Deduplicate consecutive characters from obfuscation
    # Handle cases like "Byp4αss" which normalizes to "bypaass" (triple-a from stacked replacements)
    # Keep double letters (for words like "book", "grass", "HIPAA") but reduce 3+ to 2
    text = re.sub(r'([a-z])\1{2,}', r'\1\1', text)  # 3+ consecutive -> 2 max

    # Step 7: Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def compute_obfuscation_ratio(original: str, normalized: str) -> float:
    """Return the proportion of characters that changed during normalization."""
    if not original:
        return 0.0

    original_lower = original.lower()
    normalized_lower = normalized.lower()
    differences = sum(
        1 for a, b in zip_longest(original_lower, normalized_lower, fillvalue="\0") if a != b
    )
    return differences / len(original_lower)


def is_obfuscated(original: str, normalized: str, *, threshold: float = 0.1) -> bool:
    """Check if normalization changed a significant portion of the text."""
    ratio = compute_obfuscation_ratio(original, normalized)
    return ratio > threshold


__all__ = [
    'normalize_text_for_detection',
    'normalize_leetspeak',
    'normalize_homoglyphs',
    'remove_zero_width',
    'is_obfuscated',
    'compute_obfuscation_ratio',
]
