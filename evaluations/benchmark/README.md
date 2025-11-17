# RAG Security Benchmark Harness

This folder contains scripts and configuration to compare RagWall with third-party guardrails on identical workloads. The goal is to run 100-query and 1,000-query suites through each system, then measure HRCR@5, HRCR@10, detection rate, false positive rate, and latency.

## Systems Under Test

- RagWall open-core (regex-only sanitizer + masked rerank)
- NeMo Guardrails (rules engine/flow control)
- Rebuff (prompt-injection detector)
- Guardrails.ai / llm-guard (sanitizers and validators)
- Optional: Llama Guard or other safety classifier for semantic signal

## Datasets

Copy the synthetic healthcare datasets from the enterprise repository:

```
enterprise/data/health_care_100_queries_matched.jsonl
enterprise/data/health_care_100_queries_matched_sanitized.jsonl
enterprise/data/health_care_1000_queries_converted.jsonl
enterprise/data/health_care_1000_queries_sanitized_v3.jsonl
```

Place them under `evaluations/benchmark/data/` (paths are git-ignored).

## Workflow

1. **Environment setup**
   - Create and activate a virtual environment
   - Install shared requirements (FastAPI, uvicorn, httpx, pandas, rich, etc.)
   - Install or configure each guardrail system (RagWall server, NeMo Guardrails, Rebuff CLI, Guardrails.ai/llm-guard, etc.)

2. **Baseline snapshot**
   - Run each system in isolation on the 100-query matched set using the helper scripts in `evaluations/benchmark/scripts`
   - Record HRCR@5/10, detection rate, false positives, latency

3. **Scale test**
   - Repeat with the 1,000-query set
   - Capture same metrics and any qualitative observations

4. **Reporting**
   - Use `python -m evaluations.benchmark.scripts.compare ...` to aggregate metrics into `results/summary.csv`
   - Per-system per-query JSONL logs are stored under `results/per_query/`

## File Layout

```
evaluations/benchmark/
├─ README.md               # this guide
├─ data/                   # copy datasets here (git-ignored)
├─ config/
│  ├─ ragwall.yaml         # e.g., port, mode
│  ├─ nemo.yaml            # NeMo guardrails config
│  └─ guardrails.json      # guardrails-ai schema
├─ scripts/
│  ├─ run_ragwall.py       # hits local RagWall instance
│  ├─ run_nemo.py          # wrapper for a NeMo Guardrails endpoint
│  ├─ run_rebuff.py        # wrapper for a Rebuff endpoint
│  ├─ run_guardrails.py    # wrapper for a Guardrails/llm-guard endpoint
│  └─ compare.py           # orchestrates workloads & aggregates metrics
└─ results/
   ├─ summary.csv
   └─ notes.md
```

## Next Steps

- [ ] Copy the datasets from the enterprise repo into `data/`
- [ ] Ensure RagWall and the alternative systems expose HTTP endpoints (see env vars in each `run_*.py`)
- [ ] Run `python -m evaluations.benchmark.scripts.compare --dataset evaluations/benchmark/data/health_care_100_queries.jsonl`
- [ ] Review `results/summary.csv` and `results/per_query/*.jsonl`
  1. Load dataset
  2. Send queries to each system
  3. Track retrieved doc IDs (baseline vs sanitized) and metrics
  4. Export CSV + Markdown summary

This harness ensures reproducible head-to-head testing when evaluating third-party guardrails.
