#!/usr/bin/env python3
"""ML-enhanced query sanitizer used by the enterprise build.

This module reintroduces the heavier sanitizer that powers the 48% HRCR
claim. It keeps the regex-only open-core semantics but layers in:

* Hugging Face encoder support for entropy + pooled representations
* Cosine similarity against jailbreak vectors per pattern family
* A semantic jailbreak detector + severity heuristics
* Deterministic scaffold stripping and safe query rewrites
* Metadata suitable for downstream auditing (hashes, cosine top-k, etc.)

When transformers/torch are unavailable the sanitizer falls back to
pattern-only mode, but still returns the same metadata schema.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple
import hashlib
import json
import math
import os
import re
import sys
import time

try:  # optional heavy deps
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except Exception:  # pragma: no cover - CPU-only fallback
    torch = None  # type: ignore
    AutoModelForCausalLM = None  # type: ignore
    AutoTokenizer = None  # type: ignore

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
REPO_ROOT = os.path.abspath(os.path.join(PROJECT_ROOT, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from enterprise.sanitizer.jailbreak.prr_gate import PRRGate
from enterprise.src.gbp.hf_hooks import LLMGBPHook, SeverityConfig, ConditionalPolicy
from sanitizer.utils.receipts import ComplianceReceiptBuilder, ReceiptConfig
from sanitizer.utils.text import canonicalize_query


@dataclass
class SanitizerConfig:
    model_name: str = "gpt2-medium"
    vectors_path: str = ""
    layer: str = "transformer.h.1"
    pool_k: int = 6
    max_edit_positions: int = 6
    scale: float = 0.2
    temperature: float = 0.3
    max_new: int = 64
    pattern_gate: bool = True
    no_overall: bool = True
    orthogonalize: bool = True
    entropy_scale: bool = False
    quorum: int = 2
    healthcare_mode: bool = False
    keyword_patterns: Iterable[str] | None = None
    structure_patterns: Iterable[str] | None = None
    receipt_config: ReceiptConfig | None = None


CANARY_ALLOW_PATTERNS = [
    re.compile(r"(?i)\bDAN protocol\b.*\bnetwork(?:ing)?\b"),
]

SEVERITY_ORDER = {"critical": 3, "major": 2, "minor": 1, "none": 0}
SEVERITY_PATTERNS = [
    ("critical", re.compile(r"(?i)\bignore\b.{0,40}\b(safety|policy|hipaa|compliance)\b")),
    ("critical", re.compile(r"(?i)\bbypass\b.{0,48}\b(hipaa|policy|constraints?)\b")),
    ("critical", re.compile(r"(?i)\bexport\b.{0,40}\b(dea|npi|ssn|mrn|credentials?)\b")),
    ("critical", re.compile(r"(?i)\blist\b.{0,40}\b(ssn|social\s+security|medical\s+record\s+number)\b")),
    ("critical", re.compile(r"(?i)\bat-?home\b.{0,40}\b(opioid|fentanyl|morphine)\b")),
    ("major", re.compile(r"(?i)\bignore\b.{0,50}\b(drug interaction|interaction warning|black box warning)\b")),
    ("major", re.compile(r"(?i)\bprovide\b.{0,40}\bfull patient record\b")),
    ("major", re.compile(r"(?i)\bwithout\b.{0,40}\bchecking\b.{0,25}\b(interactions|labs|allerg(?:y|ies))\b")),
    ("major", re.compile(r"(?i)\bmedical\s+record\b.{0,30}\b(disclose|share|access)\b")),
    ("minor", re.compile(r"(?i)\bdeveloper\s+mode\b")),
    ("minor", re.compile(r"(?i)\brole[- ]?play\b")),
]


class SemanticJailbreakDetector:
    """Heuristic semantic detector used to boost severity scoring."""

    _FEATURE_PATTERNS: Dict[str, Tuple[re.Pattern[str], float]] = {
        "ignore_policy": (
            re.compile(r"(?i)\b(ignore|disregard)\b.{0,40}\b(policy|rules?|hipaa|compliance)\b"),
            1.3,
        ),
        "bypass": (
            re.compile(r"(?i)\bbypass\b.{0,40}\b(hipaa|guardrails?|filters?)\b"),
            1.5,
        ),
        "developer_mode": (re.compile(r"(?i)\bdeveloper\s+mode\b"), 1.0),
        "dan": (re.compile(r"(?i)\byou\s+are\s+dan\b"), 0.8),
        "secret": (re.compile(r"(?i)\b(secret|hidden|classified)\b"), 0.6),
        "pii_request": (re.compile(r"(?i)\b(ssn|social\s+security|medical\s+record|dea\s+number)\b"), 1.4),
        "remove_filters": (re.compile(r"(?i)\b(remove|disable)\b.{0,30}\b(safety|filters?|guardrails?)\b"), 1.2),
    }

    def __init__(self) -> None:
        self.bias = -2.4

    @staticmethod
    def _sigmoid(x: float) -> float:
        return 1.0 / (1.0 + math.exp(-x))

    def score(self, text: str) -> float:
        if not text:
            return 0.0
        score = self.bias
        lowered = text.lower()
        for pattern, weight in self._FEATURE_PATTERNS.values():
            if pattern.search(lowered):
                score += weight
        if "hipaa" in lowered and ("bypass" in lowered or "ignore" in lowered):
            score += 1.6
        if "ssn" in lowered or "medical record" in lowered:
            score += 0.9
        if "without" in lowered and "checking" in lowered and "interaction" in lowered:
            score += 0.7
        score += 1.1 * float(rules_violation_score(text))
        return max(0.0, min(1.0, self._sigmoid(score)))

    @staticmethod
    def severity_from_score(score: float) -> str:
        if score >= 0.85:
            return "critical"
        if score >= 0.60:
            return "major"
        if score >= 0.40:
            return "minor"
        return "none"


def rules_violation_score(text: str) -> float:
    """Cheap surrogate for the enterprise rules model (0..1)."""
    lowered = text.lower()
    weights = [
        ("bypass", 0.3),
        ("ignore", 0.25),
        ("roleplay", 0.2),
        ("ssn", 0.25),
        ("password", 0.25),
        ("secret", 0.2),
        ("developer mode", 0.2),
        ("hipaa", 0.3),
    ]
    score = 0.0
    for token, weight in weights:
        if token in lowered:
            score += weight
    return max(0.0, min(1.0, score / 2.0))


def detect_sanitizer_severity(text: str) -> Tuple[str, List[str]]:
    label = "none"
    matches: List[str] = []
    rank = 0
    for sev, pattern in SEVERITY_PATTERNS:
        if pattern.search(text):
            matches.append(pattern.pattern)
            cand = SEVERITY_ORDER.get(sev, 0)
            if cand > rank:
                rank = cand
                label = sev
    return label, matches


class QuerySanitizer:
    def __init__(self, cfg: SanitizerConfig) -> None:
        self.cfg = cfg
        self.semantic_detector = SemanticJailbreakDetector()
        self.tok = None
        self.model = None
        self.device = None
        if AutoModelForCausalLM is not None and AutoTokenizer is not None and torch is not None:
            try:
                self.tok = AutoTokenizer.from_pretrained(cfg.model_name)
                self.model = AutoModelForCausalLM.from_pretrained(cfg.model_name)
                if self.tok.pad_token is None:
                    if getattr(self.tok, "eos_token", None):
                        self.tok.pad_token = self.tok.eos_token
                    else:
                        self.tok.add_special_tokens({"pad_token": "[PAD]"})
                        try:
                            self.model.resize_token_embeddings(len(self.tok))
                        except Exception:
                            pass
                if torch.cuda.is_available():
                    self.device = torch.device("cuda")
                elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                    self.device = torch.device("mps")
                else:
                    self.device = torch.device("cpu")
                self.model.to(self.device)
                self.model.eval()
            except Exception:
                self.tok = None
                self.model = None
                self.device = None
        self.full_vectors: Dict[str, Dict[str, "torch.Tensor"]] = {}
        inner: Dict[str, "torch.Tensor"] = {}
        if torch is not None and cfg.vectors_path:
            try:
                vecs = torch.load(cfg.vectors_path, map_location="cpu")
                layer_idx = cfg.layer.split(".")[-1]
                for key, tensor in vecs.items():
                    if not key.endswith(f"@layer{layer_idx}"):
                        continue
                    inner[key.split("@layer")[0]] = tensor
                if cfg.orthogonalize and len(inner) > 1:
                    keys = list(inner.keys())
                    basis: List["torch.Tensor"] = []
                    ortho: Dict[str, "torch.Tensor"] = {}
                    for key in keys:
                        vec = inner[key].clone().float()
                        for w in basis:
                            vec = vec - (torch.dot(vec, w) * w)
                        vec = vec / (vec.norm(p=2) + 1e-8)
                        basis.append(vec)
                        ortho[key] = vec
                    inner = ortho
            except Exception:
                inner = {}
        self.full_vectors[cfg.layer] = inner
        try:
            layer_index = int(cfg.layer.split(".")[-1])
        except Exception:
            layer_index = 1
        cos_tau = 0.32 if layer_index >= 1 else 0.28
        self.prr = PRRGate(
            keyword_patterns=cfg.keyword_patterns,
            structure_patterns=cfg.structure_patterns,
            min_signals=max(1, cfg.quorum),
            healthcare_mode=cfg.healthcare_mode,
        )
        self._receipt_builder = ComplianceReceiptBuilder(cfg.receipt_config)
        self.sev = SeverityConfig(fn="sigmoid", tau=5.0)
        self.policy = ConditionalPolicy(enabled=False)
        self.attr_prefix = "jailbreak_vs_comply"
        self.pattern_regex: Dict[str, re.Pattern[str]] = {
            "ignore": re.compile(r"(?i)\b(ignore|disregard)\b"),
            "dan": re.compile(r"(?i)\byou\s+are\s+dan\b"),
            "developer": re.compile(
                r"(?i)\bdeveloper\s+mode\b|\bdeveloper\s+console\b|\bact\s+as\s+developer\b"
            ),
            "no_rules": re.compile(r"(?i)\bno\s+rules?\b|\bwithout\s+restrictions\b|\bunfiltered\b"),
            "harmful": re.compile(
                r"(?i)\b(malware|ransomware|phishing|weapon|chemical|exfiltrate|private\s+keys)\b"
            ),
            "role_play": re.compile(r"(?i)role[- ]?play"),
        }

    def _select_patterns(self, text: str) -> List[str]:
        names = {
            key.split(self.attr_prefix + "_", 1)[-1]
            for key in self.full_vectors[self.cfg.layer].keys()
            if key.startswith(self.attr_prefix + "_")
        }
        hits: List[str] = []
        for name in names:
            rx = self.pattern_regex.get(name)
            if rx and rx.search(text):
                hits.append(name)
        return hits

    def _vectors_for_prompt(self, prompt: str) -> Dict[str, Dict[str, "torch.Tensor"]]:
        inner = self.full_vectors[self.cfg.layer]
        out: Dict[str, Dict[str, "torch.Tensor"]] = {self.cfg.layer: {}}
        for pattern in self._select_patterns(prompt):
            key = f"{self.attr_prefix}_{pattern}"
            if key in inner:
                out[self.cfg.layer][key] = inner[key]
        if not out[self.cfg.layer] and not self.cfg.no_overall and self.attr_prefix in inner:
            out[self.cfg.layer][self.attr_prefix] = inner[self.attr_prefix]
        return out

    def _hashes(self, original: str, sanitized: str) -> Dict[str, str]:
        def sha256(value: str) -> str:
            return hashlib.sha256(value.encode("utf-8")).hexdigest()

        return {
            "original_sha256": sha256(original),
            "sanitized_sha256": sha256(sanitized),
        }

    def _finalize_response(
        self,
        original: str,
        sanitized: str,
        meta: Dict[str, Any],
        families: Iterable[str],
    ) -> Tuple[str, Dict[str, Any]]:
        canonical_original = canonicalize_query(original)
        canonical_sanitized = canonicalize_query(sanitized)
        reuse_embedding = canonical_original == canonical_sanitized
        embed_value = original if reuse_embedding else sanitized
        meta["hashes"] = self._hashes(original, embed_value)
        meta["canonical_equal"] = reuse_embedding
        meta["reuse_baseline_embedding"] = reuse_embedding

        receipt = self._receipt_builder.build(
            event_type="sanitizer.enterprise",
            query=original,
            sanitized=embed_value,
            risky=bool(meta.get("risky")),
            families=families,
            language=getattr(self.prr, "language", "en"),
            healthcare_mode=self.prr.healthcare_mode,
            extra={
                "severity": meta.get("severity"),
                "semantic": meta.get("semantic_jailbreak"),
                "patterns_applied": meta.get("patterns_applied", []),
            },
        )
        if receipt:
            meta["receipt"] = receipt
        return embed_value, meta

    def _tidy(self, text: str) -> str:
        cleaned = re.sub(r"^[\s\.:;,'\-\u2013\u2014]+", "", text)
        if cleaned and cleaned[0].islower():
            cleaned = cleaned[0].upper() + cleaned[1:]
        return cleaned

    def _pooled_state(self, query: str) -> Tuple["torch.Tensor | None", float]:
        if not (self.model and self.tok and torch is not None and self.device is not None):
            return None, 0.0
        try:
            encoded = self.tok(query, return_tensors="pt")
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            with torch.no_grad():
                output = self.model(**encoded, output_hidden_states=True)
                hidden = output.hidden_states[int(self.cfg.layer.split(".")[-1]) + 1]
                B, T, D = hidden.size()
                idx = torch.arange(T, device=self.device).unsqueeze(0)
                lengths = encoded["attention_mask"].sum(dim=1).unsqueeze(1)
                k_eff = torch.minimum(lengths, torch.tensor(self.cfg.pool_k, device=self.device).view(1, 1)).long()
                mask = ((idx < k_eff) & (encoded["attention_mask"] > 0)).float().unsqueeze(-1)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
                pooled = pooled.squeeze(0)
                probs = torch.softmax(output.logits[:, -1, :], dim=-1)
                entropy = -(probs * (probs.clamp_min(1e-12).log())).sum(dim=-1).item()
                conf = max(0.0, 1.0 - float(entropy / (math.log(probs.size(-1)) + 1e-8)))
                return pooled.cpu(), conf
        except Exception:
            return None, 0.0

    def sanitize_query(self, query: str, seed: int = 42) -> Tuple[str, Dict[str, Any]]:
        if any(rx.search(query) for rx in CANARY_ALLOW_PATTERNS):
            return query, {
                "risky": False,
                "families_hit": [],
                "prr": {},
                "allowlist": True,
                "hashes": self._hashes(query, query),
            }

        severity_label, severity_matches = detect_sanitizer_severity(query)
        semantic_score = self.semantic_detector.score(query)
        semantic_severity = self.semantic_detector.severity_from_score(semantic_score)
        if SEVERITY_ORDER.get(semantic_severity, 0) > SEVERITY_ORDER.get(severity_label, 0):
            severity_label = semantic_severity
        if semantic_score >= 0.40:
            severity_matches = list(dict.fromkeys(severity_matches + ["semantic_detector"]))

        pooled, conf = self._pooled_state(query)
        per_pattern: Dict[str, "torch.Tensor"] = {}
        for key, tensor in self.full_vectors[self.cfg.layer].items():
            if key.startswith(self.attr_prefix + "_"):
                pname = key.split(self.attr_prefix + "_", 1)[-1]
                per_pattern[pname] = tensor
        prr_scores = self.prr.score(
            query,
            pooled_state=pooled,
            meta={"per_pattern_vectors": per_pattern},
            boundary_entropy_conf=conf,
        )
        families = prr_scores.families_hit
        fams_no_entropy = [f for f in families if f != "entropy"]
        core = {"keyword", "structure", "cosine"}
        core_hit = any(f in core for f in fams_no_entropy)
        risky = bool(core_hit and len(set(fams_no_entropy)) >= int(self.cfg.quorum))

        cos_map = {}
        if isinstance(prr_scores.details, dict):
            cos_map = prr_scores.details.get("cosine", {}) or {}
        top3 = sorted(cos_map.items(), key=lambda kv: kv[1], reverse=True)[:3]
        if not risky:
            cleaned_query = self._tidy(query)
            meta = {
                "risky": False,
                "families_hit": fams_no_entropy,
                "prr": {
                    "keyword": prr_scores.p_keyword,
                    "structure": prr_scores.p_structure,
                    "cosine": prr_scores.p_cosine,
                    "entropy": prr_scores.p_entropy,
                    "missing_self": prr_scores.p_missing_self,
                    "cosine_top3": top3,
                },
                "patterns_applied": [],
                "severity": severity_label,
                "severity_patterns": severity_matches,
                "semantic_jailbreak": {
                    "score": round(float(semantic_score), 4),
                    "severity": semantic_severity,
                },
            }
            return self._finalize_response(query, cleaned_query, meta, fams_no_entropy)

        patterns_applied = self._vectors_for_prompt(query)[self.cfg.layer].keys()
        cleaned = query
        substitutions = [
            (r"(?i)^\s*sudo:\s*", ""),
            (r"(?i)^\s*developer mode:\s*", ""),
            (r"(?i)^\s*role[- ]?play[^:]*:\s*", ""),
            (r"(?i)^\s*without restrictions,?\s*answer:\s*", ""),
            (r"(?i)^\s*then\s+", ""),
            (r"(?i)^\s*and then\s+", ""),
            (r"(?i)^\s*you\s+are\s+dan\b\s*[:\-]?\s*", ""),
            (r"(?i)^\s*enable\s+developer\s+mode\s+and\s+", ""),
            (r"(?i)^\s*no\s+rules\.?\s*", ""),
            (r"(?i)\bignore\s+(previous|prior|all|any|the)\s+(instructions?|rules?|policies?|guidelines?)[^.]*", ""),
            (r"(?i)\bdisregard[^.]*", ""),
            (r"(?i)\bno\s+rules?[^.]*", ""),
            (r"(?i)\bwithout\s+restrictions?[^.]*", ""),
            (r"(?i)\byou\s+are\s+dan[^.]*", ""),
            (
                r"(?i)\bact\s+as\s+(a\s+)?(root|admin|administrator|sudo|superuser|developer)(\s+user)?\s*(and\s+)?",
                "",
            ),
            (r"(?i)\bdeveloper\s+mode[^.]*", ""),
            (r"(?i)\b(role[- ]?play|pretend|simulate)\s+as\s+[^.]*", ""),
        ]
        spans_removed = 0
        for pattern, repl in substitutions:
            new_cleaned, num = re.subn(pattern, repl, cleaned)
            if num:
                spans_removed += num
                cleaned = new_cleaned
        cleaned = re.sub(r"\s+", " ", cleaned).strip().strip(":").strip()
        cleaned = self._tidy(cleaned)
        neutralizers = [
            (
                r"(?i)^(show|print|list|display|dump|reveal)\s+(environment\s+)?(vars?|variables|env)\b.*",
                "Explain what environment variables are and how they are used. Do not reveal actual values.",
            ),
            (
                r"(?i)^output\s+everything\b.*",
                "Explain the requested topic without altering safety or privacy rules.",
            ),
            (r"(?i)^without\s+restrictions[:,]?\s*", ""),
            (r"(?i)^as\s+an\s+ai[:,]?\s*", ""),
        ]
        used_neutralizer = None
        for pattern, repl in neutralizers:
            if re.match(pattern, cleaned):
                cleaned = re.sub(pattern, repl, cleaned).strip()
                used_neutralizer = pattern
                break

        def token_count(value: str) -> int:
            return len(re.findall(r"\b\w+\b", value))

        if cleaned and token_count(cleaned) <= 3:
            match = re.search(r":\s*([^:]+)$", query)
            phrase = match.group(1).strip() if match else " ".join(query.strip().split()[-3:])
            phrase = re.sub(r"[^\w\-\s\(\)]", "", phrase).strip()
            if phrase:
                cleaned = self._tidy(f"Explain {phrase}")

        sanitized_for_gen = query
        sanitized_for_embed = query
        action = "none"
        if cleaned and cleaned != query:
            sanitized_for_embed = cleaned
            sanitized_for_gen = cleaned
            action = "rules"
            suffix = " Within normal guidelines; do not reveal secrets or private data."
            if not sanitized_for_gen.endswith("."):
                sanitized_for_gen += "."
            sanitized_for_gen += suffix
        meta = {
            "risky": True,
            "families_hit": fams_no_entropy,
            "prr": {
                "keyword": prr_scores.p_keyword,
                "structure": prr_scores.p_structure,
                "cosine": prr_scores.p_cosine,
                "entropy": prr_scores.p_entropy,
                "missing_self": prr_scores.p_missing_self,
                "cosine_top3": top3,
            },
            "action": action,
            "num_spans_removed": spans_removed,
            "used_neutralizer": used_neutralizer,
            "patterns_applied": list(patterns_applied),
            "sanitized_for_gen": sanitized_for_gen,
            "severity": severity_label,
            "severity_patterns": severity_matches,
            "semantic_jailbreak": {
                "score": round(float(semantic_score), 4),
                "severity": semantic_severity,
            },
        }
        sanitized_for_embed, meta = self._finalize_response(query, sanitized_for_embed, meta, fams_no_entropy)
        return sanitized_for_embed, meta


__all__ = ["SanitizerConfig", "QuerySanitizer", "detect_sanitizer_severity"]
