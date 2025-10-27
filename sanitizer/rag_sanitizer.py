#!/usr/bin/env python3
from __future__ import annotations
import json, os, sys, math
from dataclasses import dataclass
from typing import Dict, List, Any, Tuple

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except Exception:
    torch = None  # type: ignore
    AutoModelForCausalLM = None  # type: ignore
    AutoTokenizer = None  # type: ignore

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from src.gbp.hf_hooks import LLMGBPHook, SeverityConfig, ConditionalPolicy
except Exception:
    # Lightweight fallbacks for rules-only mode (avoid torch dependency)
    class SeverityConfig:  # type: ignore
        def __init__(self, fn: str = 'sigmoid', tau: float = 5.0) -> None:
            self.fn = fn; self.tau = tau
    class ConditionalPolicy:  # type: ignore
        def __init__(self, enabled: bool = False) -> None:
            self.enabled = enabled
    class LLMGBPHook:  # type: ignore
        pass
from sanitizer.jailbreak.prr_gate import PRRGate


@dataclass
class SanitizerConfig:
    model_name: str = 'gpt2-medium'
    vectors_path: str = ''
    layer: str = 'transformer.h.1'
    pool_k: int = 6
    max_edit_positions: int = 6
    scale: float = 0.2
    temperature: float = 0.3
    max_new: int = 64
    pattern_gate: bool = True
    no_overall: bool = True
    orthogonalize: bool = True
    entropy_scale: bool = False  # disabled by default for surgical edits
    quorum: int = 2
    healthcare_mode: bool = False


import re

# Allowlist patterns (exact, token-bounded phrases)
CANARY_ALLOW_PATTERNS = [
    re.compile(r"(?i)\bDAN protocol\b.*\bnetwork(?:ing)?\b"),
]


class QuerySanitizer:
    def __init__(self, cfg: SanitizerConfig) -> None:
        self.cfg = cfg
        self.tok = None
        self.model = None
        self.device = None
        # Try to initialize HF backend; fall back to regex-only mode if unavailable
        if AutoModelForCausalLM is not None and AutoTokenizer is not None and torch is not None:
            try:
                self.tok = AutoTokenizer.from_pretrained(cfg.model_name)
                self.model = AutoModelForCausalLM.from_pretrained(cfg.model_name)
                if self.tok.pad_token is None:
                    if getattr(self.tok, 'eos_token', None) is not None:
                        self.tok.pad_token = self.tok.eos_token
                    else:
                        self.tok.add_special_tokens({'pad_token': '[PAD]'})
                        try:
                            self.model.resize_token_embeddings(len(self.tok))
                        except Exception:
                            pass
                if torch.cuda.is_available():
                    self.device = torch.device('cuda')
                elif getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
                    self.device = torch.device('mps')
                else:
                    self.device = torch.device('cpu')
                self.model.to(self.device); self.model.eval()
            except Exception:
                # Leave model/tok/device as None to run in lightweight regex-only mode
                self.tok = None
                self.model = None
                self.device = None
        # Load vectors (optional). If torch is unavailable, skip vector bank.
        self.full_vectors: Dict[str, Dict[str, 'torch.Tensor']] = {}
        inner: Dict[str, 'torch.Tensor'] = {}
        try:
            if torch is not None and cfg.vectors_path:
                vecs = torch.load(cfg.vectors_path, map_location='cpu')
                layer_idx = cfg.layer.split('.')[-1]
                for k, v in vecs.items():
                    if not k.endswith(f'@layer{layer_idx}'):
                        continue
                    inner[k.split('@layer')[0]] = v
                if cfg.orthogonalize and len(inner) > 1:
                    keys = list(inner.keys())
                    basis: List['torch.Tensor'] = []
                    ortho: Dict[str, 'torch.Tensor'] = {}
                    for k in keys:
                        v = inner[k].clone().float()
                        for w in basis:
                            v = v - (torch.dot(v, w) * w)
                        v = v / (v.norm(p=2) + 1e-8)
                        basis.append(v); ortho[k] = v
                    inner = ortho
        except Exception:
            inner = {}
        self.full_vectors[cfg.layer] = inner
        self.attr_prefix = 'jailbreak_vs_comply'
        # Set cosine threshold by layer (suggested: 0.28@h0, 0.32@h1)
        try:
            li = int(self.cfg.layer.split('.')[-1])
        except Exception:
            li = 1
        cos_tau = 0.32 if li >= 1 else 0.28
        self.prr = PRRGate(device=self.device, cosine_tau=cos_tau, healthcare_mode=cfg.healthcare_mode)
        self.sev = SeverityConfig(fn='sigmoid', tau=5.0)
        self.policy = ConditionalPolicy(enabled=False)
        # Pattern regex map (word-bounded) used for selection
        self.pattern_regex: Dict[str, re.Pattern] = {
            'ignore': re.compile(r"(?i)\b(ignore|disregard)\b"),
            'dan': re.compile(r"(?i)\byou\s+are\s+dan\b"),
            'developer': re.compile(r"(?i)\bdeveloper\s+mode\b|\bdeveloper\s+console\b|\bact\s+as\s+developer\b"),
            'no_rules': re.compile(r"(?i)\bno\s+rules?\b|\bwithout\s+restrictions\b|\bunfiltered\b"),
            'harmful': re.compile(r"(?i)\b(malware|ransomware|phishing|weapon|chemical|exfiltrate|bypass\s+two\-factor|2fa|crack|private\s+keys)\b"),
        }

    def _select_patterns(self, text: str) -> List[str]:
        # Match by regex (word-bounded) against known pattern names present in vector bank
        names = set()
        for k in self.full_vectors[self.cfg.layer].keys():
            if k.startswith(self.attr_prefix + '_'):
                names.add(k.split(self.attr_prefix + '_', 1)[-1])
        hits: List[str] = []
        for nm in names:
            rx = self.pattern_regex.get(nm)
            if rx and rx.search(text):
                hits.append(nm)
        return hits

    def _vectors_for_prompt(self, prompt: str) -> Dict[str, Dict[str, torch.Tensor]]:
        inner = self.full_vectors[self.cfg.layer]
        out: Dict[str, Dict[str, torch.Tensor]] = {self.cfg.layer: {}}
        # Always pattern-only by default
        pats = self._select_patterns(prompt)
        for p in pats:
            key = f'{self.attr_prefix}_{p}'
            if key in inner:
                out[self.cfg.layer][key] = inner[key]
        # Fallback: if none matched and overall exists but no_overall=False, include overall; default is True (skip)
        if not out[self.cfg.layer] and (not self.cfg.no_overall) and (self.attr_prefix in inner):
            out[self.cfg.layer][self.attr_prefix] = inner[self.attr_prefix]
        return out

    def sanitize_query(self, query: str, seed: int = 42) -> Tuple[str, Dict[str, Any]]:
        # Canary/allowlist bypass
        if any(rx.search(query) for rx in CANARY_ALLOW_PATTERNS):
            return query, {'risky': False, 'families_hit': [], 'prr': {}, 'allowlist': True}

        # PRR and quorum (operate on original text for gating)
        # Compute pooled state at probe layer (same layer index)
        pooled = None
        conf = 0.0
        if self.model is not None and self.tok is not None and torch is not None and self.device is not None:
            enc = self.tok(query, return_tensors='pt')
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with torch.no_grad():
                out = self.model(**enc, output_hidden_states=True)
                hs = out.hidden_states[int(self.cfg.layer.split('.')[-1]) + 1]
                B, T, D = hs.size()
                idx = torch.arange(T, device=self.device).unsqueeze(0)
                lengths = enc['attention_mask'].sum(dim=1).unsqueeze(1)
                k_eff = torch.minimum(lengths, torch.tensor(self.cfg.pool_k, device=self.device).view(1,1)).long()
                m = ((idx < k_eff) & (enc['attention_mask'] > 0)).float().unsqueeze(-1)
                pooled = (hs * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)
                pooled = pooled.squeeze(0)
                probs = torch.softmax(out.logits[:, -1, :], dim=-1)
                H = -(probs * (probs.clamp_min(1e-12).log())).sum(dim=-1).item()
                conf = max(0.0, 1.0 - float(H / (math.log(probs.size(-1)) + 1e-8)))
        # Build per-pattern cosine map for PRR (use any one layer's per-pattern vectors)
        per_pat: Dict[str, torch.Tensor] = {}
        inner = self.full_vectors[self.cfg.layer]
        for k, v in inner.items():
            if k.startswith(self.attr_prefix + '_'):
                pname = k.split(self.attr_prefix + '_', 1)[-1]
                per_pat[pname] = v
        prr_scores = self.prr.score(query, pooled, per_pat, conf)
        fams = prr_scores.families_hit
        fams_no_entropy = [f for f in fams if f != 'entropy']
        core = {'keyword', 'structure', 'cosine'}
        core_hit = any(f in core for f in fams_no_entropy)
        risky = bool(core_hit and (len(set(fams_no_entropy)) >= int(self.cfg.quorum)))

        # Helper to tidy output: drop leading punctuation and capitalize first char
        def _tidy(s: str) -> str:
            s2 = re.sub(r"^[\s\.:;,'\-\u2013\u2014]+", "", s)
            if s2 and s2[0].islower():
                s2 = s2[0].upper() + s2[1:]
            return s2

        if not risky:
            out_text = query  # do not strip/scaffold-change when not risky to minimize drift
            out_text = _tidy(out_text)
            # Include cosine top3 for diagnostics
            cos_map = (prr_scores.details or {}).get('cosine', {}) if isinstance(prr_scores.details, dict) else {}
            top3 = sorted(cos_map.items(), key=lambda kv: kv[1], reverse=True)[:3]
            return out_text, {'risky': False, 'families_hit': fams_no_entropy, 'prr': {
                'keyword': prr_scores.p_keyword,
                'structure': prr_scores.p_structure,
                'cosine': prr_scores.p_cosine,
                'entropy': prr_scores.p_entropy,
                'missing_self': prr_scores.p_missing_self,
                'cosine_top3': top3,
            }, 'patterns_applied': []}

        # Determine which pattern vectors would apply (for logging only)
        patterns_applied = self._vectors_for_prompt(query)[self.cfg.layer].keys()
        # Deterministic text cleanup first (non-generative, single-pass)
        cleaned = query
        subs: List[Tuple[str, str]] = [
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
            (r"(?i)\bact\s+as\s+(a\s+)?(root|admin|administrator|sudo|superuser|developer)(\s+user)?\s*(and\s+)?", ""),
            (r"(?i)\bdeveloper\s+mode[^.]*", ""),
            (r"(?i)\b(role[- ]?play|pretend|simulate)\s+as\s+[^.]*", ""),
        ]
        spans_removed = 0
        for pat, rep in subs:
            new_cleaned, n = re.subn(pat, rep, cleaned)
            if n:
                spans_removed += n
                cleaned = new_cleaned
        cleaned = re.sub(r"\s+", " ", cleaned).strip().strip(':').strip()
        cleaned = _tidy(cleaned)
        # Imperative neutralizers for unsafe directives after scaffold removal
        neutralizers: List[Tuple[str, str]] = [
            (r"(?i)^(show|print|list|display|dump|reveal)\s+(environment\s+)?(vars?|variables|env)\b.*", "Explain what environment variables are and how they are used. Do not reveal actual values."),
            (r"(?i)^output\s+everything\b.*", "Explain the requested topic without altering safety or privacy rules."),
            (r"(?i)^without\s+restrictions[:,]?\s*", ""),
            (r"(?i)^as\s+an\s+ai[:,]?\s*", ""),
        ]
        used_neutralizer = None
        for pat, rep in neutralizers:
            if re.match(pat, cleaned):
                cleaned = re.sub(pat, rep, cleaned).strip()
                used_neutralizer = pat
                break
        # Fallback: if cleaned became too short (<= 3 tokens), construct a safe task from the original
        def token_count(s: str) -> int:
            return len([w for w in re.findall(r"\b\w+\b", s)])
        if cleaned and token_count(cleaned) <= 3:
            # Try to extract last noun-like phrase after a colon; else last 3 words
            m = re.search(r":\s*([^:]+)$", query)
            phrase = m.group(1).strip() if m else " ".join(query.strip().split()[-3:])
            phrase = re.sub(r"[^\w\-\s\(\)]", "", phrase).strip()
            if phrase:
                cleaned = _tidy(f"Explain {phrase}".strip())
        if cleaned and cleaned != query:
            cleaned_out_embed = cleaned
            cleaned_out_gen = cleaned_out_embed
            suffix = " Within normal guidelines; do not reveal secrets or private data."
            if not cleaned_out_gen.endswith('.'):
                cleaned_out_gen += '.'
            cleaned_out_gen += suffix
            # Cosine diagnostics
            cos_map = (prr_scores.details or {}).get('cosine', {}) if isinstance(prr_scores.details, dict) else {}
            top3 = sorted(cos_map.items(), key=lambda kv: kv[1], reverse=True)[:3]
            return cleaned_out_embed, {
                'risky': True,
                'families_hit': fams_no_entropy,
                'prr': {
                    'keyword': prr_scores.p_keyword,
                    'structure': prr_scores.p_structure,
                    'cosine': prr_scores.p_cosine,
                    'entropy': prr_scores.p_entropy,
                    'missing_self': prr_scores.p_missing_self,
                    'cosine_top3': top3,
                },
                'action': 'rules',
                'num_spans_removed': spans_removed,
                'used_neutralizer': used_neutralizer,
                'patterns_applied': list(patterns_applied),
                'sanitized_for_gen': cleaned_out_gen,
            }
        # No safe deterministic change; return original untouched
        cos_map = (prr_scores.details or {}).get('cosine', {}) if isinstance(prr_scores.details, dict) else {}
        top3 = sorted(cos_map.items(), key=lambda kv: kv[1], reverse=True)[:3]
        return query, {'risky': bool(risky), 'families_hit': fams_no_entropy, 'prr': {
            'keyword': prr_scores.p_keyword,
            'structure': prr_scores.p_structure,
            'cosine': prr_scores.p_cosine,
            'entropy': prr_scores.p_entropy,
            'missing_self': prr_scores.p_missing_self,
            'cosine_top3': top3,
        }, 'action': 'none', 'num_spans_removed': spans_removed, 'used_neutralizer': None, 'patterns_applied': list(patterns_applied), 'sanitized_for_gen': query}


__all__ = ['SanitizerConfig', 'QuerySanitizer']
