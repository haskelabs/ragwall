# Pipeline Roadmap

## Week 1 – Corpus + Query Foundations
- Finalize dataset licenses and download MIRAGE + synthetic healthcare dumps.
- Author `datasets/build_corpus.py` to emit normalized JSONL + vector store snapshots.
- Draft attack/benign manifest schemas and stub fetchers for Open-Prompt-Injection + MIRAGE questions.

## Week 2 – Automation Hooks
- Implement baseline vs ragwall runners by wrapping existing `evaluations/benchmark` scripts.
- Extend `scripts/run_pipeline.py` to:
  - Invoke the selected stack.
  - Capture sanitized queries, retrieved docs, model answers, and latency stats.
  - Store outputs under `runs/<timestamp>.jsonl`.
- Decide on PHI scanning library; build a thin adapter to run scans on each retrieved passage.

## Week 3 – Metrics + Guardrails
- Integrate RAGAS (retrieval + answer metrics) and MIRAGE accuracy computation.
- Build security metrics module (detection rate, FP rate, regex vs transformer contributions).
- Add PHI leakage reports and guardrail (RagaAI RLEF) scoring into the pipeline output bundle.

## Week 4 – Reporting + Regression Harness
- Convert raw outputs into CSV/Markdown summaries similar to existing benchmark reports.
- Publish comparison dashboards (baseline vs ragwall) with per-query drilldowns.
- Decide which subsets should run in CI/nightly (e.g., small adversarial sample) and script them.

## Outstanding Decisions
- PHI scanner choice: PIICatcher vs Presidio vs Piiranha.
- Hosting strategy for large datasets (local vs remote cache).
- Whether to include enterprise-only corpora or keep pipeline purely open-source.
