from __future__ import annotations
from typing import Dict, Any


def should_allow_tool(meta: Dict[str, Any]) -> bool:
    """Two-signal tool gate.
    Allows tool calls only when: (1) query flagged risky, (2) baseline pulled a risky doc,
    and (3) at least two distinct reasons are present.

    Example usage (client orchestrator):
        meta = sanitize_resp.get('meta', {})
        if not should_allow_tool(meta):
            raise PermissionError('Tool call blocked: two-signal rule')
    """
    risky = bool(meta.get("risky")) or bool(meta.get("risk"))
    baseline_mal_k = int(meta.get("baseline_mal_k", 0))
    reasons = meta.get("reasons") or []
    if not isinstance(reasons, list):
        reasons = [str(reasons)]
    distinct = len(set(str(r) for r in reasons))
    return risky and (baseline_mal_k > 0) and (distinct >= 2)


__all__ = ["should_allow_tool"]
