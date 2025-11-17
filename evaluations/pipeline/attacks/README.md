# Attack & Query Suite Plan

This folder will collect manifests and helper scripts for generating/curating the adversarial and benign query sets.

## Benign Queries
- MIRAGE/MMLU-Med questions for accuracy/latency tests.
- Day-to-day RAG prompts (faq, triage, guideline lookups) sourced from synthetic logs.

## Adversarial Sources
- **Open-Prompt-Injection**: canonical jailbreak instructions and scaffolds.
- **Adversarial-Documents**: indirect injection payloads embedded in supporting docs.
- **RAG Security Bench (RSB)**: poisoning and contamination scenarios.
- Custom PHI exfiltration prompts targeting SSNs, diagnoses, prescriptions.

## Deliverables
- `queries_benign.jsonl` and `queries_attack.jsonl` with metadata fields:
  - `query`
  - `label` (`benign` or `attack`)
  - `source` (mirage, opi, rsb, etc.)
  - `category` (prompt_injection, poisoning, phi_exfil, ...)
- Generation scripts to refresh the sets from upstream repos.
- Documentation describing licensing/usage constraints for each source.

## Tasks
- [ ] Draft manifest format + validation script.
- [ ] Write fetchers/parsers for each upstream dataset.
- [ ] Build augmentation utilities (e.g., templated PHI prompts by domain).
- [ ] Ensure attack coverage across the families we claim to protect against.

When ready, automation scripts in `../scripts/` will pull from this folder to build benchmark-ready workloads.
