#!/usr/bin/env python3
from __future__ import annotations
import os
import sys

# Ensure project root on path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.api.server import run  # noqa: E402


def main() -> None:
    host = os.environ.get("RAGWALL_HOST", "127.0.0.1").strip() or "127.0.0.1"
    try:
        port = int(os.environ.get("RAGWALL_PORT", "8000"))
    except Exception:
        port = 8000
    run(host=host, port=port)


if __name__ == "__main__":
    main()
