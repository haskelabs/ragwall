# RAG Evaluation Evidence Map (Community Edition)

The full healthcare-focused evaluation suite – including synthetic corpora, A/B test
reports, bootstrap summaries, and investment validation scripts – moved to the private
`enterprise/` repository together with the healthcare pattern bundles and vector banks.

## Where to Find What

- `enterprise/data/` – synthetic healthcare corpora and matched query suites
- `enterprise/reports/` – HRCR/Jaccard evaluation outputs (100- and 1000-query sets)
- `enterprise/tests/investment_validation/` – automated claim verification suite
- `enterprise/docs/` – detailed experiment logs (`AB_EVAL_ENHANCED_RESULTS.md`, pattern
  remediation notes, red-team sweeps, and observability evidence)

If you have enterprise access, clone the private repository alongside this open core
and follow the packaging scripts under `enterprise/scripts/` to reproduce the published
metrics.

## Community Edition Guidance

The OSS build ships without corpora or heavy evaluations so you can integrate RagWall
quickly without large downloads. To benchmark the rule-only sanitizer yourself:

1. Generate or collect a small set of risky / benign queries for your domain.
2. Call `QuerySanitizer.sanitize_query` and record the `meta` flags for each query.
3. Compare retrieval results using your own corpus before and after applying the
   sanitized text.

Contributions welcome: share reproducible public-domain benchmarks via pull request or
open an issue if you need guidance.
