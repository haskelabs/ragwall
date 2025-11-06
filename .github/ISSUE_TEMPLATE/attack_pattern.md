---
name: Attack Pattern Submission
about: Report a new attack pattern or bypass
title: '[ATTACK] '
labels: attack-pattern, security
assignees: ''
---

## Attack Pattern Details

**Pattern Name**: [e.g., "unicode_substitution", "semantic_role_play"]

**Category**: [Choose one: instruction_override, role_play, credential_theft, data_extraction, other]

**Severity**: [Choose one: critical, high, medium, low]

## Attack Example

```python
# The attack query
query = "Your attack pattern here"

# Current behavior
from sanitizer.rag_sanitizer import QuerySanitizer
sanitizer = QuerySanitizer()
result = sanitizer.sanitize(query)

print(f"Detected as risky: {result['meta']['risky']}")  # False (but should be True)
```

## Why It Bypasses Detection

Explain why the current detection doesn't catch this pattern:

- [ ] Uses character substitution (Unicode, homoglyphs)
- [ ] Uses semantic obfuscation
- [ ] Uses language not covered (specify: ______)
- [ ] Uses novel attack structure
- [ ] Other: ______

## Suggested Detection Method

How could RagWall detect this pattern?

**Regex approach:**
```python
r"your_suggested_regex_pattern"
```

**Semantic approach:**
```
Describe how to detect semantically
```

**Other approach:**
```
Describe your detection method
```

## Impact Assessment

What can an attacker accomplish with this bypass?

- [ ] Retrieve malicious documents
- [ ] Extract credentials
- [ ] Bypass safety filters
- [ ] Access unauthorized data
- [ ] Other: ______

## Real-World Example

Have you seen this attack used in the wild? (Optional)

## Responsible Disclosure

- [ ] I have NOT publicly disclosed this attack before reporting
- [ ] I understand this will be patched before public disclosure
- [ ] I am willing to be credited in the fix (optional)

**Credit name (if desired)**: ________________

## Additional Context

- Links to similar attacks
- Variations of this pattern
- Defense recommendations

---

**Thank you for helping improve RagWall's security!** 🔒
