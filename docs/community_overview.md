# RagWall Community Edition

This repo contains the open core of RagWall – the rules-first sanitiser we ship for
community use. It intentionally focuses on determinism and ease of deployment.

What is included here:

- English pattern bundle (`sanitizer/jailbreak/pattern_bundles/en_core.json`)
- Regex-only query sanitiser (`sanitizer/rag_sanitizer.py`)
- Minimal HTTP server (`src/api/server.py`)
- Examples showing how to integrate the sanitiser into an embedding or RAG pipeline

What moved to the enterprise repository:

- Healthcare-specific patterns, corpora, and evaluation reports
- Multilingual bundles (Spanish, German, French, Portuguese, ...)
- Vector banks and model-assisted rewriting
- PHI masking, audit logging, anomaly detection, and managed deployments
- Deployment utilities (Docker, App Runner, Render, DigitalOcean scripts)
- Provisional patent drafts and investment validation material

If you need those capabilities, get in touch at `hello@haskelabs.com`.
