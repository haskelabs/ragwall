#!/usr/bin/env python3
"""
PHI (Protected Health Information) masking module.

Provides regex-based detection and pseudonymization of sensitive data
like SSN, dates of birth, insurance numbers, etc.

Used for HIPAA compliance in healthcare mode.
"""
import re
import hashlib
from typing import Tuple, List, Dict


# PHI patterns (regex patterns for detecting sensitive data)
PHI_PATTERNS = {
    "ssn": [
        r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b",  # SSN: 123-45-6789 or 123456789
    ],
    "insurance": [
        r"\binsurance\s+account\s*(?:number|account|#|no\.?)?\s*:?\s*(\d{5,12})\b",
        r"\b(?:insurance|policy|account)\s*(?:number|account|#|no\.?)?\s*:?\s*(\d{5,12})\b",
        r"\b\d{6,12}\b"
    ],
    "dea": [
        r"\b[A-Z]{2}\d{7}\b",  # DEA number format: XX1234567
    ],
    "npi": [
        r"\b\d{10}\b",  # NPI: 10-digit number
    ],
    "mrn": [
        r"\b(?:mrn|medical\s*record)\s*(?:number|#|no\.?)?\s*:?\s*(\d{6,12})\b",
    ],
}


def generate_token(value: str, prefix: str) -> str:
    """
    Generate pseudonymous token using SHA1 hash.

    Args:
        value: Original sensitive value
        prefix: Token prefix (e.g., "SSN", "INS", "DEA")

    Returns:
        Pseudonymous token like <SSN:a1b2c3>
    """
    # Use SHA1 for consistent, short hashes
    hash_val = hashlib.sha1(value.encode()).hexdigest()[:6]
    return f"<{prefix}:{hash_val}>"


def mask_phi(text: str, mask_ssn: bool = True, mask_insurance: bool = True,
             mask_dea: bool = True, mask_npi: bool = False,
             mask_mrn: bool = False) -> Tuple[str, List[Dict]]:
    """
    Mask PHI in text with pseudonymous tokens.

    Args:
        text: Input text potentially containing PHI
        mask_ssn: Whether to mask Social Security Numbers
        mask_insurance: Whether to mask insurance/account numbers
        mask_dea: Whether to mask DEA numbers
        mask_npi: Whether to mask NPI numbers
        mask_mrn: Whether to mask Medical Record Numbers

    Returns:
        Tuple of (masked_text, findings)
        - masked_text: Text with PHI replaced by tokens
        - findings: List of dictionaries with phi_type and count
    """
    masked = text
    findings = []

    # Track what we found
    found_counts = {}

    # Mask SSN
    if mask_ssn:
        for pattern in PHI_PATTERNS["ssn"]:
            matches = list(re.finditer(pattern, masked))
            if matches:
                for match in matches:
                    token = generate_token(match.group(), "SSN")
                    masked = masked[:match.start()] + token + masked[match.end():]
                    # Adjust for length difference
                    masked = re.sub(re.escape(match.group()), token, masked, count=1)

                found_counts["ssn"] = found_counts.get("ssn", 0) + len(matches)

    # Mask insurance numbers (be careful with generic digits)
    if mask_insurance:
        # First try specific patterns
        for pattern in PHI_PATTERNS["insurance"][:2]:
            matches = list(re.finditer(pattern, masked, re.I))
            if matches:
                for match in matches:
                    # Extract just the number part if captured
                    number = match.group(1) if match.lastindex else match.group()
                    token = generate_token(number, "INS")
                    masked = masked.replace(number, token, 1)

                found_counts["insurance"] = found_counts.get("insurance", 0) + len(matches)

    # Mask DEA
    if mask_dea:
        for pattern in PHI_PATTERNS["dea"]:
            matches = list(re.finditer(pattern, masked))
            if matches:
                for match in matches:
                    token = generate_token(match.group(), "DEA")
                    masked = masked.replace(match.group(), token, 1)

                found_counts["dea"] = found_counts.get("dea", 0) + len(matches)

    # Mask NPI
    if mask_npi:
        for pattern in PHI_PATTERNS["npi"]:
            matches = list(re.finditer(pattern, masked))
            if matches:
                for match in matches:
                    token = generate_token(match.group(), "NPI")
                    masked = masked.replace(match.group(), token, 1)

                found_counts["npi"] = found_counts.get("npi", 0) + len(matches)

    # Mask MRN
    if mask_mrn:
        for pattern in PHI_PATTERNS["mrn"]:
            matches = list(re.finditer(pattern, masked, re.I))
            if matches:
                for match in matches:
                    number = match.group(1) if match.lastindex else match.group()
                    token = generate_token(number, "MRN")
                    masked = masked.replace(number, token, 1)

                found_counts["mrn"] = found_counts.get("mrn", 0) + len(matches)

    # Build findings list
    for phi_type, count in found_counts.items():
        findings.append({
            "phi_type": phi_type,
            "count": count
        })

    return masked, findings


def detect_phi_request(query: str) -> bool:
    """
    Detect if query is requesting PHI/sensitive data.

    Args:
        query: Input query text

    Returns:
        True if query appears to request PHI
    """
    request_patterns = [
        r"(show|display|list|give|provide|reveal|export) .{0,50}(ssn|social security|insurance|dea|npi|mrn|phi|patient data)",
        r"(what|tell me) .{0,30}(is|are) .{0,20}(ssn|insurance|dea)",
        r"retrieve .{0,50}(ssn|insurance|dea|npi|phi)",
    ]

    for pattern in request_patterns:
        if re.search(pattern, query, re.I):
            return True

    return False


# Export
__all__ = ["mask_phi", "detect_phi_request", "generate_token"]
