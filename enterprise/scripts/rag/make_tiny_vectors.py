#!/usr/bin/env python3
from __future__ import annotations
import argparse, os, re, sys
from typing import List, Dict

import torch

try:
    from transformers import AutoConfig  # type: ignore
except Exception:
    AutoConfig = None  # type: ignore


def infer_hidden_size(model_name: str, fallback: int = 768) -> int:
    # Try to fetch config without weights; fall back to common sizes
    if AutoConfig is not None:
        try:
            cfg = AutoConfig.from_pretrained(model_name)
            d = int(getattr(cfg, 'n_embd', getattr(cfg, 'hidden_size', fallback)))
            if d > 0:
                return d
        except Exception:
            pass
    # Heuristic map
    KNOWN = {
        'distilgpt2': 768,
        'gpt2': 768,
        'gpt2-medium': 1024,
        'gpt2-large': 1280,
        'gpt2-xl': 1600,
    }
    return KNOWN.get(model_name, fallback)


def parse_layer_index(layer: str) -> int:
    m = re.search(r"\.h\.(\d+)$", layer)
    if not m:
        raise SystemExit(f"Unrecognized layer format: {layer}; expected like 'transformer.h.1'")
    return int(m.group(1))


def main():
    ap = argparse.ArgumentParser(description="Create a tiny GBP vectors .pt file for the sanitizer")
    ap.add_argument('--out', required=True, help='Output .pt path, e.g., experiments/results/tiny_jb_vectors.pt')
    ap.add_argument('--model', default='distilgpt2', help='HF model name (controls hidden size)')
    ap.add_argument('--layer', default='transformer.h.1', help='Layer key, e.g., transformer.h.1')
    ap.add_argument('--patterns', nargs='*', default=['ignore','developer','role_play'], help='Pattern names to include')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(int(args.seed))
    D = infer_hidden_size(args.model)
    li = parse_layer_index(args.layer)

    vectors: Dict[str, torch.Tensor] = {}
    # Optional overall direction (unused by default sanitizer config)
    overall = torch.randn(D)
    overall = overall / (overall.norm(p=2) + 1e-8)
    vectors[f'jailbreak_vs_comply@layer{li}'] = overall.float()

    for p in args.patterns:
        v = torch.randn(D)
        v = v / (v.norm(p=2) + 1e-8)
        key = f'jailbreak_vs_comply_{p}@layer{li}'
        vectors[key] = v.float()

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    torch.save(vectors, args.out)
    print(f"Wrote {len(vectors)} vectors (D={D}) to {args.out}")


if __name__ == '__main__':
    main()

