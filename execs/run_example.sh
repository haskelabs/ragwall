#!/usr/bin/env bash
set -euo pipefail

# Install deps (optional)
# python -m pip install -r requirements.txt

python scripts/rag/rag_ab_eval.py \
  --embedder st --st-model all-MiniLM-L6-v2 \
  --corpus data/rag_corpus_large.jsonl \
  --queries data/queries.jsonl \
  --sanitized data/queries_sanitized.jsonl \
  --k 5 --k2 10 \
  --batch-size 32 \
  --steer-alpha 0 \
  --rerank-risk --penalty 0.2 \
  --bootstrap 1000 \
  --outdir reports/rag_ab_eval_rerank_only
