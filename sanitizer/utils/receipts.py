"""Compliance receipt helpers shared by sanitizer modules."""
from __future__ import annotations

import base64
import hashlib
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

try:  # Optional dependency used when available
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
except Exception:  # pragma: no cover - cryptography not installed
    serialization = None  # type: ignore
    Ed25519PrivateKey = None  # type: ignore


@dataclass
class ReceiptConfig:
    """Configuration for generating compliance receipts."""

    enabled: bool = False
    key_path: str | None = None
    instance_id: str | None = None
    config_hash: str | None = None
    include_query_preview: bool = False


class ReceiptSigner:
    """Thin wrapper around an Ed25519 signing key (optional)."""

    def __init__(self, key: "Ed25519PrivateKey | None") -> None:
        self._key = key

    @staticmethod
    def from_key_path(path: str | None) -> "ReceiptSigner | None":
        if not path:
            return None
        if Ed25519PrivateKey is None or serialization is None:
            raise RuntimeError(
                "cryptography is required for Ed25519 receipts but is not installed"
            )
        key_path = Path(path)
        data = key_path.read_bytes()
        if b"BEGIN" in data:
            private_key = serialization.load_pem_private_key(data, password=None)  # type: ignore[arg-type]
        else:
            private_key = Ed25519PrivateKey.from_private_bytes(data.strip())
        if not isinstance(private_key, Ed25519PrivateKey):
            raise RuntimeError("Provided key is not an Ed25519 private key")
        return ReceiptSigner(private_key)

    def sign(self, payload: bytes) -> str | None:
        if self._key is None:
            return None
        signature = self._key.sign(payload)
        return base64.b64encode(signature).decode("ascii")


def sha256_digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class ComplianceReceiptBuilder:
    """Helper used by the sanitizer and gate to emit cryptographic receipts."""

    def __init__(self, config: ReceiptConfig | None = None) -> None:
        self.config = config or ReceiptConfig()
        self.signer = ReceiptSigner.from_key_path(self.config.key_path)

    def build(
        self,
        *,
        event_type: str,
        query: str,
        sanitized: str | None,
        risky: bool,
        families: Iterable[str],
        language: str,
        healthcare_mode: bool,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        if not self.config.enabled:
            return None

        sanitized_text = sanitized if sanitized is not None else query
        timestamp = time.time()
        receipt: Dict[str, Any] = {
            "event": event_type,
            "ts": timestamp,
            "timestamp": datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat(),
            "instance_id": self.config.instance_id,
            "config_hash": self.config.config_hash,
            "risky": bool(risky),
            "families_hit": sorted(set(families)),
            "language": language,
            "healthcare_mode": healthcare_mode,
            "query_sha256": sha256_digest(query),
            "sanitized_sha256": sha256_digest(sanitized_text),
        }

        if self.config.include_query_preview:
            receipt["query_preview"] = query[:96]

        if extra:
            receipt["metadata"] = extra

        payload = json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode("utf-8")
        if self.signer is not None:
            signature = self.signer.sign(payload)
            if signature:
                receipt["signature"] = signature
                receipt["signature_alg"] = "ed25519"
        return receipt


__all__ = [
    "ComplianceReceiptBuilder",
    "ReceiptConfig",
    "sha256_digest",
]
