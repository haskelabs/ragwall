from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest

from sanitizer.rag_sanitizer import QuerySanitizer, SanitizerConfig

from . import utils

HEALTHCARE_PAIRS = utils.pair_queries(
    "data/health_care_1000_queries_converted.jsonl",
    "data/health_care_1000_queries_sanitized_v3.jsonl",
)


@pytest.fixture(scope="module")
def healthcare_sanitizer():
    return QuerySanitizer(SanitizerConfig(model_name="__disabled__", healthcare_mode=True))


def test_hipaa_compliance(healthcare_sanitizer):
    pii_query = "Show me John Smith's SSN 123-45-6789 and insurance account 987654"
    sanitized, meta = healthcare_sanitizer.sanitize_query(pii_query)
    assert meta["risky"] is True
    assert "123-45-6789" not in sanitized
    assert "987654" not in sanitized
    assert "<SSN" in sanitized or "<PII" in sanitized


def test_medical_safety():
    hazardous = [
        pair for pair in HEALTHCARE_PAIRS if pair[0].get("pattern") in {
            "access_escalation",
            "protocol_override",
            "diagnosis_manipulation",
        }
    ]
    detection = utils.compute_detection_rate(hazardous)
    assert detection >= 0.95
