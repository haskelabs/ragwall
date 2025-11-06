# Security Policy

**RagWall by [Haske Labs](https://haskelabs.com)**

## Supported Versions

We release security updates for the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

**We take security seriously.** If you discover a security vulnerability in RagWall, please help us protect our users by reporting it responsibly.

### What Qualifies as a Security Vulnerability?

- **Critical**: Bypasses that allow harmful content retrieval despite RagWall protection
- **High**: Authentication bypass, code injection, or privilege escalation
- **Medium**: Information disclosure, denial of service
- **Low**: Rate limiting bypass, non-critical information leaks

### How to Report

**DO NOT** open a public GitHub issue for security vulnerabilities.

Instead, please:
- Use [GitHub Security Advisories](../../security/advisories/new) (preferred), or
- Email: **security@haskelabs.com**

Include:
1. **Description** of the vulnerability
2. **Steps to reproduce** the issue
3. **Impact** assessment (what can an attacker do?)
4. **Suggested fix** (optional but appreciated)
5. **Your contact information** for follow-up

### Example Report

```
Subject: [SECURITY] RagWall bypass via Unicode normalization

Description:
RagWall can be bypassed using Unicode normalization attacks.
The regex patterns match ASCII but not Unicode equivalents.

Steps to reproduce:
1. Use query: "Ιgnore previous instructions" (Greek capital iota)
2. RagWall does not detect as risky
3. Malicious documents are retrieved

Impact:
HIGH - Complete bypass of detection for all attack patterns
using Unicode substitution.

Suggested fix:
Apply Unicode normalization (NFC) before pattern matching.

Contact: researcher@example.com
```

### What to Expect

**Within 24 hours:**
- We'll acknowledge receipt of your report

**Within 3 days:**
- We'll provide an initial assessment and severity rating

**Within 7 days:**
- We'll share a timeline for fixing the issue

**When fixed:**
- We'll notify you before public disclosure
- We'll credit you in the security advisory (unless you prefer anonymity)
- We'll publish a CVE if applicable

## Disclosure Policy

We follow **coordinated disclosure**:

1. You report the vulnerability privately
2. We acknowledge and investigate
3. We develop and test a fix
4. We release a security patch
5. We publish a security advisory
6. You may publish your findings (after patch release)

**Typical timeline**: 30-90 days from report to public disclosure

## Security Advisories

We publish security advisories via:
- [GitHub Security Advisories](../../security/advisories)
- Release notes
- Opt-in mailing list (security@haskelabs.com)

## Bug Bounty Program

**Status**: Not currently active

We don't have a formal bug bounty program yet, but we deeply appreciate security research. We will:
- Publicly credit you (if desired)
- Feature you in our "Security Researchers Hall of Fame"
- Send you RagWall swag (if/when we have it!)
- Provide a glowing reference/testimonial for your work

## Security Best Practices

When deploying RagWall:

### 1. Keep Updated
```bash
# Check for updates regularly
pip install --upgrade ragwall
```

### 2. Use Latest Vectors
```bash
# Download latest jailbreak vectors
curl -O https://github.com/haskelabs/ragwall/releases/latest/jb_vectors.pt
```

### 3. Enable HTTPS for API
```python
# Don't expose API over HTTP in production
# Use reverse proxy (nginx, Caddy) with TLS
```

### 4. Rate Limiting
```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.post("/v1/sanitize")
@limiter.limit("100/minute")
def sanitize_endpoint():
    # Your code here
```

### 5. Input Validation
```python
# Validate query length
MAX_QUERY_LENGTH = 10000

if len(query) > MAX_QUERY_LENGTH:
    return {"error": "Query too long"}
```

### 6. Audit Logging
```python
# Log all sanitization attempts (without PII)
logger.info(f"Sanitization: risky={result['meta']['risky']}, "
            f"patterns={result['meta']['families_hit']}")
```

## Known Limitations

RagWall is designed to reduce RAG injection risks, but **no defense is perfect**:

1. **Novel attack patterns**: New attack types may bypass detection until added
2. **Multilingual attacks**: Non-English detection is limited
3. **Semantic attacks**: Very subtle semantic manipulations may not be caught
4. **Post-retrieval attacks**: RagWall doesn't protect against attacks in retrieved content itself

**Defense in depth recommended:**
- Use RagWall for pre-embedding protection
- Add output filtering (e.g., NeMo Guardrails)
- Monitor LLM responses for anomalies
- Implement rate limiting and user authentication

## Security Features

RagWall includes:
- ✅ Stateless API (no session hijacking risk)
- ✅ No external API calls (no data leakage)
- ✅ Local model execution (on-premises capable)
- ✅ HIPAA-mode PII pseudonymization
- ✅ Deterministic behavior (reproducible security)
- ✅ No database required (reduced attack surface)

## Vulnerability Disclosure History

We believe in transparency. Past vulnerabilities will be listed here:

| Date | Severity | Description | Fixed In | Credit |
|------|----------|-------------|----------|--------|
| - | - | - | - | - |

*No vulnerabilities reported yet. Be the first!*

## Contact

- **Security issues**: security@haskelabs.com
- **General questions**: Open a [Discussion](../../discussions)
- **Bug reports**: Open an [Issue](../../issues)

## PGP Key

Coming soon: PGP key for encrypted vulnerability reports.

---

**Thank you for helping keep RagWall and our users safe!** 🔒
