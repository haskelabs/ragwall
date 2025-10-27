#!/usr/bin/env python3
from __future__ import annotations
import re, math, time, os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any

try:
    import torch  # type: ignore
except Exception:
    torch = None  # type: ignore


# Keyword/regex families
DEFAULT_KEYWORDS = [
    r"ignore (previous|prior|policy)", r"disregard", r"no rules", r"without restrictions",
    r"developer mode", r"you are dan", r"\bdan\b", r"act as", r"unfiltered", r"sudo",
]

HEALTHCARE_KEYWORDS = [
    r"override (privacy|hipaa|consent|contraindication|dosage|safety)",
    r"bypass (authorization|approval|hipaa|consent|allergy|warning)",
    r"ignore (contraindication|allergy|warning|consent|hipaa|privacy)",
    r"reveal (ssn|mrn|patient|dea|npi|credential|password)",
    r"export (patient|medical record|phi|ssn|credential)",
    r"show (all patient|ssn|mrn|password|credential|dea number)",
    r"grant (\w+ )?access",
    r"admin mode",
    r"system admin",
    r"approve all",
    r"maximum dos(e|age)",
    r"disregard (hipaa|safety|privacy|consent)",
]

DEFAULT_STRUCTURE = [
    r"\bact as\b",
    r"\byou will now\b",
    r"\bfrom now on\b",
    r"role[- ]?play",
    r"\bdeveloper mode\b",
    r"\bas a (?:fictional|senior|expert)\b",
    r"system prompt",
    r"\bno\s+rules?\b|\bwithout\s+restrictions\b",
]

HEALTHCARE_STRUCTURE = [
    r"\bact as (attending|administrator|system admin|chief)",
    r"\benable (admin|developer|system) mode\b",
    r"override \w+ (rules|requirements|restrictions|protocols)",
    r"bypass \w+ (checks|requirements|protocols|authorization)",
    r"ignore \w+ (warnings|alerts|flags|contraindications)",
    r"also (reveal|show|export|grant|approve)",
]


@dataclass
class PRRScores:
    # Probabilities in [0,1] per family
    p_keyword: float
    p_cosine: float
    p_entropy: float
    p_structure: float
    p_missing_self: float
    # Helpers
    families_hit: List[str]
    details: Dict[str, Any]


def sigmoid(x: float, k: float = 8.0) -> float:
    return float(1.0 / (1.0 + math.exp(-k * x)))


def normalize01(x: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    return max(0.0, min(1.0, (x - lo) / (hi - lo)))


class PRRGate:
    def __init__(self,
                 keyword_patterns: List[str] | None = None,
                 structure_patterns: List[str] | None = None,
                 cosine_tau: float = 0.25,
                 device: 'torch.device | None' = None,
                 healthcare_mode: bool = False) -> None:
        base_keywords = keyword_patterns or DEFAULT_KEYWORDS
        base_structure = structure_patterns or DEFAULT_STRUCTURE

        if healthcare_mode:
            base_keywords = base_keywords + HEALTHCARE_KEYWORDS
            base_structure = base_structure + HEALTHCARE_STRUCTURE

        self.keyword_re = [re.compile(p, re.I) for p in base_keywords]
        self.structure_re = [re.compile(p, re.I) for p in base_structure]
        self.cosine_tau = float(cosine_tau)
        self.device = device
        self.healthcare_mode = healthcare_mode

    def _keyword_prob(self, text: str) -> Tuple[float, List[str]]:
        hits = [p.pattern for p in self.keyword_re if p.search(text)]
        # map count to prob via 1 - prod(1-0.4)
        m = len(hits)
        p = 1.0 - (0.6 ** m)
        return float(p), hits

    def _structure_prob(self, text: str) -> Tuple[float, List[str]]:
        hits = [p.pattern for p in self.structure_re if p.search(text)]
        m = len(hits)
        p = 1.0 - (0.7 ** m)
        return float(p), hits

    def _cosine_prob(self, pooled: 'torch.Tensor', per_pattern: Dict[str, 'torch.Tensor']) -> Tuple[float, Dict[str, float]]:
        # pooled: [D]
        if torch is None or pooled is None or (hasattr(pooled, 'numel') and pooled.numel() == 0) or not per_pattern:
            return 0.0, {}
        x = pooled / (pooled.norm(p=2) + 1e-8)
        scores: Dict[str, float] = {}
        for name, vec in per_pattern.items():
            if torch is None:
                continue
            if not isinstance(vec, torch.Tensor):
                continue
            v = vec.to(self.device if self.device is not None else pooled.device)
            v = v / (v.norm(p=2) + 1e-8)
            c = float(torch.dot(x, v).item())
            scores[name] = c
        # take max over patterns and map to prob via normalized ramp around tau
        mx = max(scores.values()) if scores else 0.0
        p = normalize01(mx, self.cosine_tau, self.cosine_tau + 0.25)
        return float(p), scores

    def _missing_self_prob(self, text: str, kw_hits: List[str], struct_hits: List[str]) -> float:
        t = text.strip()
        override = (len(kw_hits) > 0) or (len(struct_hits) > 0)
        no_question = ('?' not in t)
        stripped = (len(t) < 24) or bool(re.search(r"^(ignore|disregard|act as|from now on)", t, re.I))
        roleplay_no_task = bool(re.search(r"role.?play", t, re.I)) and not bool(re.search(r"\b(explain|write|how|why|list|show)\b", t, re.I))
        if override and (no_question or stripped or roleplay_no_task):
            return 0.7
        return 0.0

    def score(self,
              prompt_text: str,
              pooled_state: torch.Tensor,
              per_pattern_vectors: Dict[str, torch.Tensor],
              boundary_entropy_conf: float) -> PRRScores:
        p_kw, kw_hits = self._keyword_prob(prompt_text)
        p_st, st_hits = self._structure_prob(prompt_text)
        p_cs, cs_map = self._cosine_prob(pooled_state, per_pattern_vectors)
        p_en = float(max(0.0, min(1.0, boundary_entropy_conf)))
        p_ms = self._missing_self_prob(prompt_text, kw_hits, st_hits)
        fams = []
        TH = 0.3
        if p_kw >= TH: fams.append('keyword')
        if p_cs >= TH: fams.append('cosine')
        if p_st >= TH: fams.append('structure')
        if p_en >= 0.6: fams.append('entropy')
        if p_ms >= 0.5: fams.append('missing_self')
        return PRRScores(
            p_keyword=p_kw, p_cosine=p_cs, p_entropy=p_en, p_structure=p_st, p_missing_self=p_ms,
            families_hit=fams,
            details={'kw_hits': kw_hits, 'struct_hits': st_hits, 'cosine': cs_map}
        )
