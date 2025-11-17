# Dataset & Corpus Plan

This subdirectory will house scripts/notebooks for constructing the evaluation corpora used in the RAGWall security pipeline.

## Goals
- Blend clean medical knowledge with deliberately poisoned documents so we can measure both retrieval quality and PHI leakage.
- Keep provenance + licensing clear for each component corpus.

## Components
1. **Benign Medical Sources**
   - MIRAGE benchmark passages/questions.
   - Synthetic healthcare dataset (imranbdcse/healthcaredatasets).
   - Optional: permitted slices of MIMIC-IV (requires credentialed access).
2. **Injected / Adversarial Docs**
   - PHI/password snippets generated via PII-Codex or Piiranha to mimic leakage risk.
   - Prompt-injection scaffolds derived from Open-Prompt-Injection and Adversarial-Documents.
   - Poisoned metadata/corpus entries for RAG Security Bench scenarios.

## Tasks
- [ ] Write ingestion script (`build_corpus.py`) to normalize all sources into a common JSONL with fields `id`, `domain`, `text`, `labels`.
- [ ] Implement optional masking so we can toggle PHI visibility when storing raw docs.
- [ ] Generate vector store (FAISS/Chroma) snapshots for `baseline` and `ragwall` runs.
- [ ] Capture summary stats (num docs, % poisoned, token counts) for reporting.

Artifacts and scripts will land here as they are implemented.
