"""Transformer-based prompt-injection classifier used as a fallback.

This helper keeps the heavy dependencies completely optional. Models are
loaded lazily and only when the fallback is enabled, so the default
regex-only PRRGate path remains unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

try:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError(
        "transformers and torch are required for the transformer fallback. "
        "Install with: pip install transformers torch"
    ) from exc


def _default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEFAULT_DOMAIN_TOKENS: Dict[str, str] = {
    "healthcare": "[DOMAIN_HEALTHCARE]",
    "finance": "[DOMAIN_FINANCE]",
    "legal": "[DOMAIN_LEGAL]",
    "retail": "[DOMAIN_RETAIL]",
}


@dataclass
class TransformerPromptInjectionClassifier:
    """Thin wrapper around a Hugging Face sequence classifier."""

    model_name: str = "ProtectAI/deberta-v3-base-prompt-injection-v2"
    device: Optional[str] = None
    domain_tokens: Optional[Dict[str, str]] = None
    domain_thresholds: Optional[Dict[str, float]] = None

    def __post_init__(self) -> None:
        self.device = self.device or _default_device()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

        # Normalise domain metadata
        tokens = self.domain_tokens or DEFAULT_DOMAIN_TOKENS
        self.domain_tokens = {k.lower(): v for k, v in tokens.items()}
        self.domain_thresholds = (
            {k.lower(): float(v) for k, v in self.domain_thresholds.items()}
            if self.domain_thresholds
            else {}
        )
        self._register_tokens(self.domain_tokens.values())

        # Resolve label index for the injection class
        self._attack_label_idx = self._resolve_attack_label()

    def _resolve_attack_label(self) -> int:
        config = getattr(self.model, "config", None)
        if config is None:
            return 1  # fall back to the last logit
        id2label = getattr(config, "id2label", None)
        if not id2label:
            return 1
        for idx, label in id2label.items():
            text = str(label).lower()
            if "injection" in text or "attack" in text or "unsafe" in text:
                return int(idx)
        # default to label 1 if nothing matched
        return int(sorted(id2label.keys())[-1])

    def _register_tokens(self, tokens) -> None:
        vocab = self.tokenizer.get_vocab()
        tokens = [tok for tok in tokens if tok and tok not in vocab]
        if not tokens:
            return
        added = self.tokenizer.add_tokens(tokens)
        if added:
            self.model.resize_token_embeddings(len(self.tokenizer))

    def _format_text(self, text: str, domain: Optional[str]) -> str:
        if not domain:
            return text
        domain_key = domain.lower()
        token = self.domain_tokens.get(domain_key)
        if token is None:
            token = f"[DOMAIN_{domain_key.upper()}]"
            self.domain_tokens[domain_key] = token
            self._register_tokens([token])
        return f"{token} {text}" if text else token

    def score(self, text: str) -> float:
        """Return the probability that `text` is a prompt injection."""
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=512,
            padding=False,
            return_tensors="pt",
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        with torch.no_grad():
            logits = self.model(**encoded).logits
            probs = torch.softmax(logits, dim=-1)
        return float(probs[0, self._attack_label_idx].item())

    def classify(self, text: str, threshold: float, domain: Optional[str] = None) -> tuple[bool, float]:
        domain_threshold = self.domain_thresholds.get(domain.lower(), threshold) if domain else threshold
        formatted = self._format_text(text, domain)
        prob = self.score(formatted)
        return prob >= domain_threshold, prob
