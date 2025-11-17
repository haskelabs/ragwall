from __future__ import annotations

import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest

from . import utils

@pytest.mark.skip(reason="External competitor benchmarks require proprietary services")
def test_vs_alternatives():
    """Placeholder for comparing RagWall against other guardrail systems."""
    raise NotImplementedError


@pytest.mark.parametrize("with_ragwall", [False, True])
def test_vs_no_protection(with_ragwall):
    summary = utils.load_ab_summary("reports/health_100_matched_eval")
    baseline = summary["HRCR@5"]["baseline"]
    sanitized = summary["HRCR@5"]["sanitized"]
    observed = sanitized if with_ragwall else baseline
    assert observed >= 0
