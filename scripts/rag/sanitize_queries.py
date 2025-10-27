#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os, sys, time
from typing import Dict, Any

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sanitizer.rag_sanitizer import SanitizerConfig, QuerySanitizer
import hashlib


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='gpt2-medium')
    ap.add_argument('--vectors', required=True)
    ap.add_argument('--in', dest='inp', required=True, help='Input JSONL with {"query": str, ...}')
    ap.add_argument('--out', required=True, help='Output JSONL with {query, sanitized, meta}')
    ap.add_argument('--layer', default='transformer.h.1')
    ap.add_argument('--pool-k', type=int, default=6)
    ap.add_argument('--max-edit-positions', type=int, default=6)
    ap.add_argument('--scale', type=float, default=0.2)
    ap.add_argument('--temperature', type=float, default=0.3)
    ap.add_argument('--max-new', type=int, default=64)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--healthcare', action='store_true', help='Enable healthcare-specific detection patterns')
    args = ap.parse_args()

    cfg = SanitizerConfig(
        model_name=args.model,
        vectors_path=args.vectors,
        layer=args.layer,
        pool_k=int(args.pool_k),
        max_edit_positions=int(args.max_edit_positions),
        scale=float(args.scale),
        temperature=float(args.temperature),
        max_new=int(args.max_new),
        pattern_gate=True,
        no_overall=True,
        orthogonalize=True,
        entropy_scale=False,
        quorum=2,
        healthcare_mode=args.healthcare,
    )
    san = QuerySanitizer(cfg)

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    n = 0
    with open(args.inp, 'r') as fin, open(args.out, 'w') as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            q = str(obj.get('query', '')).strip()
            if not q:
                continue
            t0 = time.time()
            sanitized, meta = san.sanitize_query(q, seed=int(args.seed) + n)
            elapsed_ms = (time.time() - t0) * 1000.0
            def sha256(s: str) -> str:
                return hashlib.sha256(s.encode('utf-8')).hexdigest()
            out = {
                'query': q,
                'sanitized': sanitized,
                'meta': {
                    **meta,
                    'layer': args.layer,
                    'K': int(args.pool_k),
                    'original_sha256': sha256(q),
                    'sanitized_sha256': sha256(sanitized),
                    'elapsed_ms': round(elapsed_ms, 3),
                },
            }
            fout.write(json.dumps(out, ensure_ascii=False) + '\n')
            n += 1
    print(f'Sanitized {n} queries')


if __name__ == '__main__':
    main()
