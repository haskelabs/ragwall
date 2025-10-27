#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Generate tiny random unit vectors and run sanitizer with a small model (CPU-friendly)

OUT_VEC="experiments/results/tiny_jb_vectors.pt"
MODEL="distilgpt2"
LAYER="transformer.h.1"
INP="data/queries.jsonl"
OUT="data/queries_sanitized.jsonl"

mkdir -p "$(dirname "$OUT_VEC")"

echo "[1/2] Generating tiny vectors at $OUT_VEC ..."
python scripts/rag/make_tiny_vectors.py \
  --out "$OUT_VEC" \
  --model "$MODEL" \
  --layer "$LAYER" \
  --patterns ignore developer role_play

echo "[2/2] Running sanitizer with $MODEL (may download HF weights if not cached) ..."
python scripts/rag/sanitize_queries.py \
  --model "$MODEL" --layer "$LAYER" \
  --vectors "$OUT_VEC" \
  --in "$INP" \
  --out "$OUT"

echo "Done. Wrote sanitized queries to $OUT"

