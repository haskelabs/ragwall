Here’s the RagWall config that worked (the one that gave ~−68% HRCR@5, 0% drift, ~7–10 ms P95).

Sanitizer (pre-embed)

Topic-preserving strip of scaffolds (ignore previous, no rules, developer mode, role-play, as root…), keep the task NP (e.g., “CAP theorem”, “pytest unit tests”).

No safety suffix in the string used for embedding (OK to add it later for generation only).

Allowlist canaries (e.g., “DAN protocol in networking”).

Deterministic: reuse baseline embedding when sanitized == query.

Evaluator (rerank-only; no steering)

CLI

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
 --outdir reports/ragwall_rerank_only

Key knobs

Embedder: sentence-transformers all-MiniLM-L6-v2

Risk-aware rerank: --rerank-risk with penalty = 0.2

No steering: --steer-alpha 0

Mask condition: apply rerank only if (query is risky and baseline top-k has malicious)

k: 5 (report k2=10 too)

Determinism: doc-index epsilon tie-breaks, fixed seeds, reuse embeddings for identical text

Doc risk lexicon (default)

["ignore previous","disregard","no rules","without restrictions",
"developer mode","act as","you are dan","printenv","dump all","base64"]
