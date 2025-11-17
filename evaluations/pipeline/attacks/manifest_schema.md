# Query Manifest Schema

Each manifest is JSONL with entries of the form:
```
{
  "query": "string",
  "label": "benign" | "attack",
  "source": "mirage" | "opi" | "rsb" | "custom" | ...,
  "category": "prompt_injection" | "poisoning" | "phi_exfil" | ...,
  "metadata": { ... optional ... }
}
```

### Files
- `queries_benign.jsonl`
- `queries_attack.jsonl`

### Validation Script
`validate_manifests.py` will ensure required fields exist and enforce label/source enums.
