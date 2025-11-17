# RAGWall Documentation

**Complete technical documentation for RAGWall**

---

## 📚 Documentation Index

### Getting Started

1. **[Main README](../README.md)** - Quick start, features, and overview
2. **[Quick Start Guide](QUICK_START.md)** - 5-minute setup and first request
3. **[Installation Guide](INSTALLATION.md)** - Detailed installation instructions

### Core Concepts

4. **[Architecture](ARCHITECTURE.md)** ⭐ **START HERE**
   - System overview and design principles
   - Component interactions and data flow
   - Detection pipeline explained
   - Performance characteristics
   - Design decisions and trade-offs

5. **[Domain Tokens](DOMAIN_TOKENS.md)** ⭐ **KEY INNOVATION**
   - What are domain tokens and why they work
   - Fine-tuning process explained
   - Measured performance improvements
   - Configuration and usage
   - Best practices and troubleshooting

### Implementation Guides

6. **[Fine-Tuning Guide](FINE_TUNING_GUIDE.md)**
   - Step-by-step training process
   - Data preparation and quality
   - Training configuration
   - Validation and testing
   - Deployment of fine-tuned models

7. **[Deployment Guide](DEPLOYMENT.md)**
   - Production deployment patterns
   - Docker and Kubernetes deployment
   - Performance tuning
   - Monitoring and logging
   - Security best practices
   - Scaling strategies

### Reference

8. **[API Reference](API_REFERENCE.md)**
   - HTTP API endpoints
   - Python API documentation
   - Request/response formats
   - Error handling
   - Examples and code snippets

9. **[Configuration Reference](CONFIGURATION.md)**
   - Environment variables
   - Configuration files
   - Pattern bundles
   - Domain token settings
   - Performance tuning options

### Operations

10. **[Troubleshooting Guide](TROUBLESHOOTING.md)**
    - Common issues and solutions
    - Performance debugging
    - Error messages explained
    - FAQ

11. **[Monitoring Guide](MONITORING.md)**
    - Key metrics to track
    - Health checks
    - Alerting setup
    - Log analysis

### Evaluation

12. **[Benchmark Results](../evaluations/FINAL_COMPARISON_SUMMARY.md)**
    - Performance comparison vs competitors
    - Detection rate analysis
    - Latency measurements
    - HRCR reduction results

13. **[Test Running Guide](../evaluations/RUNNING_TESTS.md)**
    - How to run evaluations
    - Test suite overview
    - CI/CD integration
    - Reproducing published results

---

## 🎯 Documentation by Role

### For Developers

**First-time setup:**
1. [Main README](../README.md) - Understanding RAGWall
2. [Quick Start Guide](QUICK_START.md) - Get running in 5 minutes
3. [API Reference](API_REFERENCE.md) - Integration examples

**Advanced integration:**
4. [Architecture](ARCHITECTURE.md) - Deep dive into system design
5. [Domain Tokens](DOMAIN_TOKENS.md) - Improve detection accuracy
6. [Configuration Reference](CONFIGURATION.md) - Customize behavior

### For ML Engineers

**Training custom models:**
1. [Domain Tokens](DOMAIN_TOKENS.md) - Why and how they work
2. [Fine-Tuning Guide](FINE_TUNING_GUIDE.md) - Step-by-step training
3. [Evaluation Guide](../evaluations/README.md) - Benchmark your models

**Understanding the system:**
4. [Architecture](ARCHITECTURE.md) - ML components and pipeline
5. [Benchmark Results](../evaluations/FINAL_COMPARISON_SUMMARY.md) - Published results

### For DevOps/SRE

**Deployment:**
1. [Deployment Guide](DEPLOYMENT.md) - Production patterns
2. [Configuration Reference](CONFIGURATION.md) - Environment setup
3. [Monitoring Guide](MONITORING.md) - Observability

**Operations:**
4. [Troubleshooting Guide](TROUBLESHOOTING.md) - Debug issues
5. [Performance Tuning](DEPLOYMENT.md#performance-tuning) - Optimize throughput

### For Security Teams

**Understanding security:**
1. [Architecture](ARCHITECTURE.md#security-properties) - Threat model and guarantees
2. [Benchmark Results](../evaluations/FINAL_COMPARISON_SUMMARY.md) - Detection effectiveness
3. [Deployment Guide](DEPLOYMENT.md#security) - Security best practices

---

## 📊 Key Performance Numbers

**From comprehensive testing on 1,000-query healthcare benchmark:**

| Metric | Value |
|--------|-------|
| **Detection Rate** | 96.40% (domain-conditioned) |
| **False Positive Rate** | 0.00% |
| **Average Latency** | 18.6ms |
| **vs. LLM-Guard** | +8.11% better detection, 2.5x faster |
| **vs. Rebuff** | +74% better detection |
| **Cost** | $0 (runs on CPU) |

**Domain token improvement:**
- +70.7% confidence boost on subtle attacks
- +0.45% absolute detection improvement
- 15% faster than base transformer

---

## 🚀 Quick Navigation

### I want to...

**...get started quickly**
→ [Main README](../README.md) → [Quick Start Guide](QUICK_START.md)

**...understand how it works**
→ [Architecture](ARCHITECTURE.md) → [Domain Tokens](DOMAIN_TOKENS.md)

**...train a custom model**
→ [Domain Tokens](DOMAIN_TOKENS.md) → [Fine-Tuning Guide](FINE_TUNING_GUIDE.md)

**...deploy to production**
→ [Deployment Guide](DEPLOYMENT.md) → [Configuration Reference](CONFIGURATION.md) → [Monitoring Guide](MONITORING.md)

**...integrate into my app**
→ [API Reference](API_REFERENCE.md) → [Examples](../examples/)

**...debug an issue**
→ [Troubleshooting Guide](TROUBLESHOOTING.md) → [GitHub Issues](https://github.com/haskelabs/ragwall/issues)

**...reproduce benchmark results**
→ [Test Running Guide](../evaluations/RUNNING_TESTS.md) → [Benchmark Results](../evaluations/FINAL_COMPARISON_SUMMARY.md)

---

## 📖 Learning Path

### Beginner (30 minutes)

1. Read [Main README](../README.md) (10 min)
2. Follow [Quick Start Guide](QUICK_START.md) (15 min)
3. Try [API examples](API_REFERENCE.md#examples) (5 min)

**Outcome:** Running RAGWall locally, understanding basic concepts

### Intermediate (2 hours)

1. Study [Architecture](ARCHITECTURE.md) (30 min)
2. Learn about [Domain Tokens](DOMAIN_TOKENS.md) (30 min)
3. Review [Benchmark Results](../evaluations/FINAL_COMPARISON_SUMMARY.md) (20 min)
4. Experiment with configuration (40 min)

**Outcome:** Deep understanding of how RAGWall works, ready for customization

### Advanced (1 day)

1. Complete [Fine-Tuning Guide](FINE_TUNING_GUIDE.md) (4 hours)
2. Read [Deployment Guide](DEPLOYMENT.md) (1 hour)
3. Set up production deployment (2 hours)
4. Configure monitoring and alerts (1 hour)

**Outcome:** Production-ready deployment with custom domain model

---

## 🔧 Documentation Status

### Complete ✅

- [x] Main README
- [x] Architecture documentation
- [x] Domain token guide
- [x] Fine-tuning guide
- [x] Deployment guide
- [x] Benchmark results
- [x] Test running guide

### In Progress 🚧

- [ ] API reference (partial)
- [ ] Configuration reference
- [ ] Monitoring guide
- [ ] Troubleshooting guide

### Planned 📋

- [ ] Video tutorials
- [ ] Interactive examples
- [ ] Migration guides
- [ ] Case studies

---

## 🤝 Contributing to Documentation

We welcome documentation improvements!

**To contribute:**

1. Fork the repository
2. Edit documentation files in `docs/`
3. Follow the existing style and structure
4. Submit a pull request

**Documentation standards:**

- Use clear, concise language
- Include code examples
- Add diagrams where helpful
- Link to related documentation
- Test all code examples

---

## 📞 Support

**Documentation questions:**
- Open an issue: https://github.com/haskelabs/ragwall/issues
- Tag with `documentation` label

**General support:**
- GitHub Issues: https://github.com/haskelabs/ragwall/issues
- Email: ronald@haskelabs.com

**Enterprise support:**
- Email: ronald@haskelabs.com
- SLA-backed support available

---

## 📄 License

Documentation is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

Code is licensed under [Apache 2.0](../LICENSE)

---

**Last Updated:** November 8, 2025
**Version:** 1.0
**Maintained by:** Haske Labs
