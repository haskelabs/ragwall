# RAGWall Open Core Deployment Strategy

## Current State
- **Public Repo:** github.com/haskelabs/ragwall (will be)
- **Enterprise Repo:** github.com/haskelabs/ragwall-enterprise (to be created)

## Deployment Model: Separate Private Repo

### Phase 1: Prepare Public Repo (Before Launch)

1. **Remove enterprise/ from public repo:**
   ```bash
   cd /Users/rjd/Documents/ragwall
   git rm -r enterprise/
   echo "enterprise/" >> .gitignore
   git commit -m "Remove enterprise features for open source launch"
   ```

2. **Verify open source tests still pass:**
   ```bash
   pytest tests/ -v
   # Should see: 3/3 PASSED
   ```

3. **Clean up references:**
   - Remove enterprise/ from README if mentioned
   - Update documentation to focus on open source edition
   - Add "Enterprise Edition" section mentioning additional features

### Phase 2: Create Enterprise Repo (Private)

1. **Initialize enterprise repo:**
   ```bash
   cd /Users/rjd/Documents/ragwall/enterprise
   git init
   git remote add origin git@github.com:haskelabs/ragwall-enterprise.git
   ```

2. **Set up enterprise structure:**
   ```
   ragwall-enterprise/
   ├── sanitizer/              # Full enterprise sanitizer
   ├── tests/                  # Enterprise tests (31/48 passing)
   ├── docs/                   # Enterprise documentation
   ├── LICENSE.enterprise      # Commercial license
   ├── pyproject.toml          # Enterprise package config
   └── README.md              # Enterprise-specific README
   ```

3. **Create pyproject.toml for enterprise:**
   ```toml
   [project]
   name = "ragwall-enterprise"
   version = "1.0.0"
   description = "RAGWall Enterprise Edition - Multi-language, Healthcare-ready"
   license = {text = "Commercial"}
   dependencies = [
       "ragwall>=1.0.0",  # Depends on open source
       # Enterprise-specific deps
   ]
   ```

4. **Commit and push:**
   ```bash
   git add .
   git commit -m "Initial enterprise edition"
   git push -u origin main
   ```

### Phase 3: Customer Distribution

**For Open Source Users:**
```bash
pip install ragwall
```

**For Enterprise Customers:**

Option A - Private PyPI (Recommended):
```bash
# Configure private PyPI credentials
pip install ragwall-enterprise --index-url https://pypi.haskelabs.com
```

Option B - Direct GitHub Access:
```bash
# Grant customer team access to private repo
pip install git+ssh://git@github.com/haskelabs/ragwall-enterprise.git
```

Option C - License Key System:
```bash
pip install ragwall-enterprise
# Requires RAGWALL_LICENSE_KEY environment variable
```

### Phase 4: Development Workflow

**Local Development (Your Machine):**
```
/Users/rjd/Documents/ragwall/              # Open source
/Users/rjd/Documents/ragwall-enterprise/   # Enterprise (separate clone)
```

**Making Changes:**
```bash
# Open source changes:
cd /Users/rjd/Documents/ragwall
# Edit, test, commit to haskelabs/ragwall

# Enterprise changes:
cd /Users/rjd/Documents/ragwall-enterprise
# Edit, test, commit to haskelabs/ragwall-enterprise

# If open source changes affect enterprise:
# Manually sync or use git submodules
```

## Access Control

### Public Repo (haskelabs/ragwall)
- **Visibility:** Public
- **License:** Apache 2.0
- **Access:** Anyone can clone, fork, contribute
- **Contents:** Core features only (90 English patterns)

### Private Repo (haskelabs/ragwall-enterprise)
- **Visibility:** Private
- **License:** Commercial
- **Access:** Haske Labs team + enterprise customers
- **Contents:** 288 patterns, 7 languages, healthcare mode, PHI masking

### GitHub Teams Setup
```
haskelabs/ragwall-enterprise
├── Team: Haske Labs Core (Admin)
├── Team: Enterprise Customers (Read)
└── Team: Enterprise Partners (Write)
```

## Pricing Tiers

### Open Source (Free)
- 90 English patterns
- Basic jailbreak detection
- Community support
- Apache 2.0 license

### Enterprise ($25k-$100k/year)
- 288 patterns across 7 languages
- Healthcare mode (HIPAA compliant)
- Auto language detection
- PHI masking
- Priority support
- Commercial license

## Security Considerations

1. **Never commit enterprise code to public repo**
   - Double-check before pushing to haskelabs/ragwall
   - Use separate directories to prevent accidents

2. **Customer access auditing**
   - Track who has access to private repo
   - Revoke access when contracts end
   - Monitor clone/download activity

3. **License key validation**
   - Consider adding license key check to enterprise code
   - Validate on import or API calls
   - Track usage for billing

## Migration Checklist

- [ ] Remove enterprise/ from public repo
- [ ] Add enterprise/ to .gitignore
- [ ] Create haskelabs/ragwall-enterprise private repo
- [ ] Set up enterprise pyproject.toml
- [ ] Test open source installation (pip install ragwall)
- [ ] Test enterprise installation (private access)
- [ ] Update README.md (remove enterprise references)
- [ ] Create enterprise README.md
- [ ] Set up GitHub Teams for access control
- [ ] Configure CI/CD for both repos
- [ ] Document customer onboarding process

## Post-Launch

### Open Source Launch
1. Push to github.com/haskelabs/ragwall
2. Post on Hacker News, Reddit, Twitter
3. Submit to awesome-python, awesome-security lists
4. Write blog post about RAGWall

### Enterprise Sales
1. Contact potential customers
2. Offer 30-day trial (temporary repo access)
3. Onboard paid customers (grant private repo access)
4. Provide integration support

## Questions to Decide

1. **Private PyPI hosting?**
   - Use Gemfury, packagecloud, or self-hosted?
   - Or just use GitHub private repo access?

2. **License key system?**
   - Add runtime license validation?
   - Or rely on GitHub access control?

3. **Enterprise versioning?**
   - Keep in sync with open source (1.0.0)?
   - Or version independently (enterprise-1.0.0)?

## Recommended Next Steps

1. **TODAY:** Remove enterprise/ and push public repo
2. **THIS WEEK:** Set up private ragwall-enterprise repo
3. **THIS MONTH:** Configure customer access system
4. **ONGOING:** Maintain both repos in parallel
