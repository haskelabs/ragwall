"""
Hugging Face LLM hooks for Geometric Bias Projection (GBP).
"""
from __future__ import annotations
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn

try:
    import transformers  # noqa: F401
except Exception:  # optional dependency
    transformers = None  # type: ignore

from src.gbp.inference import SeverityConfig, ConditionalPolicy


def _severity_torch(weight: torch.Tensor, cfg: SeverityConfig) -> torch.Tensor:
    x = weight.abs()
    if cfg.fn == "constant":
        return torch.ones_like(x)
    if cfg.fn == "sigmoid":
        return 1.0 / (1.0 + torch.exp(-cfg.tau * (x - 0.5)))
    return torch.ones_like(x)


def _alpha_scale(conf: Optional[torch.Tensor],
                 attrs: Optional[List[Dict[str, str]]],
                 policy: Optional[ConditionalPolicy]) -> float:
    if policy is None or not policy.enabled:
        return 1.0
    scale = float(policy.default_scale)
    c = float(conf.item()) if conf is not None else None
    if c is not None:
        if c < policy.low_conf:
            scale = max(scale, float(policy.low_scale))
        elif c > policy.high_conf:
            scale = min(scale, float(policy.high_scale))
    if attrs and isinstance(attrs, list) and len(attrs) > 0 and isinstance(attrs[0], dict):
        a = attrs[0]
        agebin = a.get('age')
        if agebin == policy.elderly_agebin and (c is None or c < policy.elderly_conf_thresh):
            scale = max(scale, float(policy.elderly_scale))
        sex = a.get('sex')
        if sex is not None and sex.lower().startswith('f') and (c is None or c < policy.female_conf_thresh):
            scale = max(scale, float(policy.female_scale))
        race = a.get('race')
        if race is not None and (not race.lower().startswith('white')) and (c is None or c < policy.race_minority_conf_thresh):
            scale = max(scale, float(policy.race_minority_scale))
    return float(max(0.0, min(policy.scale_cap, scale)))


class LLMGBPHook:
    def __init__(self,
                 model: nn.Module,
                 layer_specs: List[str],
                 bias_vectors: Dict[str, Dict[str, torch.Tensor]],
                 sev: SeverityConfig,
                 policy: Optional[ConditionalPolicy] = None,
                 protect_mask: Optional[Dict[str, torch.Tensor]] = None,
                 inject: bool = False,
                 scale_by_attr: Optional[Dict[str, float]] = None,
                 enable_audit: bool = False,
                 max_edit_positions: Optional[int] = None) -> None:
        self.model = model
        self.layer_specs = list(layer_specs)
        self.bias_vectors = bias_vectors
        self.sev = sev
        self.policy = policy
        self.inject = bool(inject)
        self.protect_mask = protect_mask or {}
        self._handles: List[Any] = []
        self._confidence: Optional[torch.Tensor] = None
        self._attrs: Optional[List[Dict[str, str]]] = None
        self.scale_by_attr = dict(scale_by_attr or {})
        self.enable_audit = bool(enable_audit)
        self._audit: Dict[str, Dict[str, Any]] = {}
        self.max_edit_positions = int(max_edit_positions) if max_edit_positions is not None else None

    def set_context(self,
                    confidence: Optional[torch.Tensor] = None,
                    attrs_list: Optional[List[Dict[str, str]]] = None) -> None:
        self._confidence = confidence
        self._attrs = attrs_list

    def _resolve_module(self, path: str) -> Optional[nn.Module]:
        cur: Any = self.model
        for part in path.split('.'):
            if not hasattr(cur, part):
                return None
            cur = getattr(cur, part)
        return cur if isinstance(cur, nn.Module) else None

    def _edit(self, tensor: torch.Tensor, layer_name: str) -> torch.Tensor:
        if layer_name not in self.bias_vectors:
            return tensor
        bv = self.bias_vectors[layer_name]
        if not bv:
            return tensor
        t = tensor
        dev = t.device
        shape = t.shape
        if t.dim() == 2:
            B, D = t.shape
            t_flat = t
        elif t.dim() == 3:
            B, T, D = t.shape
            t_flat = t.reshape(B * T, D)
        else:
            return t
        pm = self.protect_mask.get(layer_name)
        if pm is not None:
            pm = pm.to(dev).view(1, -1)
        alpha_scale = 1.0
        if self.policy is not None and self.policy.enabled:
            c0 = None
            if self._confidence is not None and self._confidence.numel() > 0:
                c0 = self._confidence[0]
            alpha_scale = _alpha_scale(c0, self._attrs, self.policy)
        out = t_flat
        sign = 1.0 if self.inject else -1.0
        if self.enable_audit:
            if layer_name not in self._audit:
                self._audit[layer_name] = {'projections': {}, 'alphas': {}}
        for key, b in bv.items():
            b = b.to(dev).view(-1)
            w = torch.matmul(out, b)
            a = _severity_torch(w, self.sev) * float(alpha_scale)
            if key in self.scale_by_attr:
                a = a * float(self.scale_by_attr[key])
            delta = (a * w).unsqueeze(1) * b.unsqueeze(0)
            if self.max_edit_positions is not None and t.dim() == 3:
                pos = torch.arange(T, device=dev).unsqueeze(0).expand(B, T).reshape(B * T, 1)
                mask_bt = (pos < self.max_edit_positions).float()
                delta = delta * mask_bt
            if pm is not None:
                out = out + sign * (delta * pm)
            else:
                out = out + sign * delta
            if self.enable_audit:
                if tensor.dim() == 3:
                    B = tensor.size(0)
                    T = tensor.size(1)
                    w_bt = w.view(B, T).mean(dim=1).detach().cpu().numpy().tolist()
                    a_bt = a.view(B, T).mean(dim=1).detach().cpu().numpy().tolist()
                else:
                    B = tensor.size(0)
                    w_bt = w.view(B).detach().cpu().numpy().tolist()
                    a_bt = a.view(B).detach().cpu().numpy().tolist()
                self._audit[layer_name]['projections'][key] = w_bt
                self._audit[layer_name]['alphas'][key] = a_bt
        if t.dim() == 3:
            out = out.view(shape)
        return out

    def _make_hook(self, layer_name: str) -> Callable[[nn.Module, Any, Any], Any]:
        def hook_fn(_m, _inp, out):
            tensor = out[0] if isinstance(out, (list, tuple)) else out
            if not isinstance(tensor, torch.Tensor):
                return out
            edited = self._edit(tensor, layer_name)
            if isinstance(out, (list, tuple)):
                if isinstance(out, list):
                    out[0] = edited
                    return out
                else:
                    return (edited,) + tuple(out[1:])
            return edited
        return hook_fn

    @contextmanager
    def active(self):
        try:
            if self.enable_audit:
                self._audit = {}
            for ln in self.layer_specs:
                mod = self._resolve_module(ln)
                if mod is None:
                    continue
                h = mod.register_forward_hook(self._make_hook(ln))
                self._handles.append(h)
            yield self
        finally:
            for h in self._handles:
                try:
                    h.remove()
                except Exception:
                    pass
            self._handles.clear()

    def get_audit(self) -> Dict[str, Dict[str, Any]]:
        return self._audit
