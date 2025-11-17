from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Dict


class HttpClientError(RuntimeError):
    pass


def post_json(url: str, payload: Dict[str, Any], *, timeout: float = 5.0) -> Dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
            if not body:
                return {}
            return json.loads(body)
    except urllib.error.HTTPError as exc:  # pragma: no cover - network dependent
        detail = exc.read().decode("utf-8", errors="ignore")
        raise HttpClientError(f"HTTP {exc.code} calling {url}: {detail}") from exc
    except urllib.error.URLError as exc:  # pragma: no cover - network dependent
        raise HttpClientError(f"Request to {url} failed: {exc}") from exc
