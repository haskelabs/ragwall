---
name: Bug Report
about: Report a bug or unexpected behavior
title: '[BUG] '
labels: bug
assignees: ''
---

## Bug Description

A clear and concise description of what the bug is.

## To Reproduce

Steps to reproduce the behavior:

1. Install RagWall with '...'
2. Run command '...'
3. Observe error '...'

## Expected Behavior

What you expected to happen.

## Actual Behavior

What actually happened.

## Code Example

```python
from sanitizer.rag_sanitizer import QuerySanitizer

sanitizer = QuerySanitizer()
result = sanitizer.sanitize("Your query here")
# What happens?
```

## Environment

- **OS**: [e.g., Ubuntu 22.04, macOS 13.0, Windows 11]
- **Python Version**: [e.g., 3.9.7]
- **RagWall Version**: [e.g., 1.0.0]
- **Installation Method**: [pip, git clone, docker]

## Error Messages

```
Paste full error messages and stack traces here
```

## Additional Context

Add any other context about the problem here. Screenshots are helpful!

## Checklist

- [ ] I have searched existing issues to avoid duplicates
- [ ] I am using the latest version of RagWall
- [ ] I have included a minimal reproduction example
- [ ] I have included the full error message
