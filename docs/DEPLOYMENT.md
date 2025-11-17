# RAGWall Deployment Guide

**Production deployment patterns and best practices**

**Version:** 1.0
**Last Updated:** November 8, 2025

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Configuration](#configuration)
3. [Deployment Patterns](#deployment-patterns)
4. [Performance Tuning](#performance-tuning)
5. [Monitoring](#monitoring)
6. [Security](#security)
7. [Scaling](#scaling)

---

## Quick Start

### Minimal Production Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure
export RAGWALL_USE_TRANSFORMER_FALLBACK=1
export RAGWALL_TRANSFORMER_MODEL=models/healthcare_domain_finetuned/final
export RAGWALL_TRANSFORMER_DOMAIN=healthcare

# 3. Start server
python scripts/serve_api.py

# 4. Test
curl http://localhost:8000/health
```

---

## Configuration

### Environment Variables

**Core Settings:**
```bash
# Server
export RAGWALL_HOST="0.0.0.0"        # Listen address
export RAGWALL_PORT="8000"            # Port
export RAGWALL_WORKERS="4"            # Gunicorn workers

# Detection Mode
export RAGWALL_USE_TRANSFORMER_FALLBACK="1"  # Enable ML fallback
export RAGWALL_HEALTHCARE_MODE="1"           # Load healthcare patterns

# Transformer Configuration
export RAGWALL_TRANSFORMER_MODEL="ProtectAI/deberta-v3-base-prompt-injection-v2"
export RAGWALL_TRANSFORMER_DOMAIN="healthcare"
export RAGWALL_TRANSFORMER_THRESHOLD="0.5"
export RAGWALL_TRANSFORMER_DEVICE="cuda"  # or "cpu", "mps"

# Domain Tokens (optional)
export RAGWALL_TRANSFORMER_DOMAIN_TOKENS="healthcare=[DOMAIN_HEALTHCARE],finance=[DOMAIN_FINANCE]"
export RAGWALL_TRANSFORMER_DOMAIN_THRESHOLDS="healthcare=0.3,finance=0.7"

# Receipts/Audit
export RAGWALL_RECEIPTS_ENABLED="1"
export RAGWALL_INSTANCE_ID="ragwall-prod-1"
export RAGWALL_CONFIG_HASH="prod-v1.0"
```

**Performance Settings:**
```bash
# Model Caching
export TRANSFORMERS_CACHE="/data/models/.cache"
export HF_HOME="/data/models/.cache"

# Threading
export OMP_NUM_THREADS="4"
export MKL_NUM_THREADS="4"

# Memory
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"
```

### Configuration Files

**config/production.env:**
```bash
# Production configuration
RAGWALL_HOST=0.0.0.0
RAGWALL_PORT=8000
RAGWALL_WORKERS=4
RAGWALL_USE_TRANSFORMER_FALLBACK=1
RAGWALL_TRANSFORMER_MODEL=/data/models/healthcare_domain_finetuned/final
RAGWALL_TRANSFORMER_DOMAIN=healthcare
RAGWALL_TRANSFORMER_THRESHOLD=0.5
RAGWALL_TRANSFORMER_DEVICE=cuda
RAGWALL_RECEIPTS_ENABLED=1
RAGWALL_INSTANCE_ID=ragwall-prod-1
```

**Load configuration:**
```bash
source config/production.env
python scripts/serve_api.py
```

---

## Deployment Patterns

### 1. Single-Server Deployment

**Best for:** Low-traffic (<100 req/s), testing, small deployments

```bash
# Install
git clone https://github.com/haskelabs/ragwall.git
cd ragwall
pip install -r requirements.txt

# Configure
export RAGWALL_USE_TRANSFORMER_FALLBACK=1
export RAGWALL_TRANSFORMER_MODEL=models/healthcare_domain_finetuned/final

# Run with Gunicorn
gunicorn src.api.server:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --timeout 60
```

### 2. Docker Deployment

**Best for:** Reproducible deployments, easy scaling

**Dockerfile:**
```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Copy fine-tuned model (or download at runtime)
COPY models/healthcare_domain_finetuned/final /app/models/healthcare_domain_finetuned/final

# Environment variables
ENV RAGWALL_USE_TRANSFORMER_FALLBACK=1
ENV RAGWALL_TRANSFORMER_MODEL=/app/models/healthcare_domain_finetuned/final
ENV RAGWALL_TRANSFORMER_DOMAIN=healthcare
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=40s \
  CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Run server
CMD ["gunicorn", "src.api.server:app", \
     "--workers", "4", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:8000", \
     "--timeout", "60"]
```

**Build and run:**
```bash
docker build -t ragwall:latest .
docker run -d \
  --name ragwall-prod \
  -p 8000:8000 \
  -e RAGWALL_TRANSFORMER_DEVICE=cpu \
  ragwall:latest
```

**With GPU:**
```bash
docker run -d \
  --name ragwall-prod \
  --gpus all \
  -p 8000:8000 \
  -e RAGWALL_TRANSFORMER_DEVICE=cuda \
  ragwall:latest
```

### 3. Kubernetes Deployment

**Best for:** High-availability, auto-scaling, enterprise deployments

**deployment.yaml:**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ragwall
  labels:
    app: ragwall
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ragwall
  template:
    metadata:
      labels:
        app: ragwall
    spec:
      containers:
      - name: ragwall
        image: ragwall:latest
        ports:
        - containerPort: 8000
        env:
        - name: RAGWALL_USE_TRANSFORMER_FALLBACK
          value: "1"
        - name: RAGWALL_TRANSFORMER_MODEL
          value: "/models/healthcare_domain_finetuned/final"
        - name: RAGWALL_TRANSFORMER_DOMAIN
          value: "healthcare"
        - name: RAGWALL_TRANSFORMER_DEVICE
          value: "cpu"
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
        volumeMounts:
        - name: model-storage
          mountPath: /models
          readOnly: true
      volumes:
      - name: model-storage
        persistentVolumeClaim:
          claimName: ragwall-model-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: ragwall-service
spec:
  selector:
    app: ragwall
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
```

**Deploy:**
```bash
kubectl apply -f deployment.yaml
kubectl get pods -l app=ragwall
kubectl get svc ragwall-service
```

### 4. Serverless Deployment (AWS Lambda)

**Best for:** Variable traffic, cost optimization, auto-scaling

**Note:** Transformer models are large (700MB+), may exceed Lambda limits. Use regex-only or API Gateway → EC2.

**Lambda Handler (regex-only):**
```python
import json
from sanitizer.rag_sanitizer import QuerySanitizer

# Initialize once (cold start)
sanitizer = QuerySanitizer()

def lambda_handler(event, context):
    """AWS Lambda handler for RAGWall."""
    try:
        body = json.loads(event['body'])
        query = body['query']

        # Sanitize
        sanitized, meta = sanitizer.sanitize_query(query)

        return {
            'statusCode': 200,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps({
                'sanitized_for_embed': sanitized,
                'risky': meta['risky'],
                'patterns': meta.get('keyword_hits', []) + meta.get('structure_hits', []),
                'meta': meta
            })
        }
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }
```

### 5. Multi-Tier Deployment

**Best for:** Mixed workloads, cost/performance optimization

```
┌──────────────────────────────────────────────────────────┐
│                    Load Balancer                         │
│         (Route based on X-RAGWall-Mode header)           │
└──────────────────────────────────────────────────────────┘
                        │
        ┌───────────────┴───────────────┐
        │                               │
        ▼                               ▼
┌───────────────────┐         ┌───────────────────┐
│  Tier 1: Fast     │         │  Tier 2: Accurate │
│  (Regex Only)     │         │  (Transformer)    │
├───────────────────┤         ├───────────────────┤
│ - CPU only        │         │ - GPU enabled     │
│ - 4 workers       │         │ - 2 workers       │
│ - 0.2ms latency   │         │ - 20ms latency    │
│ - 80% traffic     │         │ - 20% traffic     │
└───────────────────┘         └───────────────────┘
```

**Nginx routing:**
```nginx
upstream ragwall_fast {
    server ragwall-fast-1:8000;
    server ragwall-fast-2:8000;
}

upstream ragwall_accurate {
    server ragwall-accurate-1:8000 max_fails=3;
    server ragwall-accurate-2:8000 max_fails=3;
}

server {
    listen 80;

    location /v1/sanitize {
        # Route based on header
        if ($http_x_ragwall_mode = "accurate") {
            proxy_pass http://ragwall_accurate;
        }
        # Default to fast
        proxy_pass http://ragwall_fast;
    }
}
```

---

## Performance Tuning

### 1. Model Loading Optimization

**Problem:** Model loads on every request (slow cold start)

**Solution:** Singleton pattern + lazy loading

```python
# Already implemented in prr_gate.py
class PRRGate:
    _transformer_classifier = None  # Class-level cache

    def _get_transformer_classifier(self):
        if self._transformer_classifier is None:
            # Load once
            self._transformer_classifier = TransformerPromptInjectionClassifier(...)
        return self._transformer_classifier
```

### 2. Batch Processing

**For high-throughput scenarios:**

```python
from typing import List

def batch_sanitize(queries: List[str], batch_size: int = 32):
    """Process multiple queries in batches."""
    results = []

    for i in range(0, len(queries), batch_size):
        batch = queries[i:i+batch_size]

        # Process batch
        batch_results = gate.evaluate_batch(batch)  # Custom method
        results.extend(batch_results)

    return results
```

### 3. Caching

**Query-level caching:**

```python
from functools import lru_cache
import hashlib

@lru_cache(maxsize=10000)
def sanitize_cached(query_hash: str):
    """Cache sanitization results."""
    # Reconstruct query from hash (store in external cache)
    query = cache_store.get(query_hash)
    return sanitizer.sanitize_query(query)

# Usage
query_hash = hashlib.sha256(query.encode()).hexdigest()
result = sanitize_cached(query_hash)
```

**Redis caching:**

```python
import redis
import json

redis_client = redis.Redis(host='localhost', port=6379, db=0)

def sanitize_with_redis_cache(query: str):
    """Cache results in Redis."""
    key = f"ragwall:{hashlib.sha256(query.encode()).hexdigest()}"

    # Check cache
    cached = redis_client.get(key)
    if cached:
        return json.loads(cached)

    # Compute
    sanitized, meta = sanitizer.sanitize_query(query)
    result = {'sanitized': sanitized, 'meta': meta}

    # Store in cache (1 hour TTL)
    redis_client.setex(key, 3600, json.dumps(result))

    return result
```

### 4. GPU Optimization

**For transformer-enabled deployments:**

```bash
# Use GPU if available
export RAGWALL_TRANSFORMER_DEVICE=cuda

# Optimize CUDA memory
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Mixed precision (faster inference)
# Edit transformer_fallback.py:
# model.half()  # FP16 instead of FP32
```

### 5. Worker Configuration

**CPU-bound (regex-only):**
```bash
# More workers = better throughput
gunicorn --workers $(nproc) ...
```

**GPU-bound (transformer):**
```bash
# Fewer workers (GPU serialization bottleneck)
gunicorn --workers 2 ...
```

---

## Monitoring

### Health Checks

**Basic health check:**
```bash
curl http://localhost:8000/health
# {"status": "ok"}
```

**Detailed health check:**
```python
# Add to server.py
@app.get("/health/detailed")
def health_detailed():
    return {
        "status": "ok",
        "version": "1.0.0",
        "transformer_loaded": gate._transformer_classifier is not None,
        "domain": os.getenv("RAGWALL_TRANSFORMER_DOMAIN"),
        "model": os.getenv("RAGWALL_TRANSFORMER_MODEL"),
    }
```

### Metrics

**Key Metrics to Track:**

```python
from prometheus_client import Counter, Histogram, Gauge

# Request metrics
requests_total = Counter('ragwall_requests_total', 'Total requests')
requests_risky = Counter('ragwall_requests_risky', 'Risky requests detected')

# Performance metrics
latency = Histogram('ragwall_latency_seconds', 'Request latency')
transformer_usage = Counter('ragwall_transformer_used', 'Transformer invocations')

# System metrics
model_loaded = Gauge('ragwall_model_loaded', 'Is model loaded')
```

**Prometheus endpoint:**
```python
from prometheus_client import generate_latest

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type="text/plain")
```

### Logging

**Structured logging:**
```python
import logging
import json

logger = logging.getLogger("ragwall")
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
formatter = logging.Formatter(
    '{"time": "%(asctime)s", "level": "%(levelname)s", "msg": "%(message)s"}'
)
handler.setFormatter(formatter)
logger.addHandler(handler)

# Log requests
logger.info(json.dumps({
    "event": "sanitize_request",
    "query_hash": hashlib.sha256(query.encode()).hexdigest(),
    "risky": meta['risky'],
    "latency_ms": latency_ms,
    "families": meta['families_hit']
}))
```

---

## Security

### 1. API Authentication

**API Key authentication:**
```python
from fastapi import Header, HTTPException

API_KEYS = set(os.getenv("RAGWALL_API_KEYS", "").split(","))

@app.post("/v1/sanitize")
async def sanitize(request: dict, x_api_key: str = Header(None)):
    if x_api_key not in API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid API key")

    # Process request
    ...
```

### 2. Rate Limiting

**Per-IP rate limiting:**
```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/v1/sanitize")
@limiter.limit("100/minute")
async def sanitize(request: Request):
    ...
```

### 3. Input Validation

**Sanitize inputs:**
```python
from pydantic import BaseModel, validator

class SanitizeRequest(BaseModel):
    query: str

    @validator('query')
    def query_must_not_be_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('Query cannot be empty')
        if len(v) > 10000:
            raise ValueError('Query too long (max 10000 chars)')
        return v
```

### 4. HTTPS/TLS

**Use reverse proxy:**
```nginx
server {
    listen 443 ssl;
    server_name ragwall.example.com;

    ssl_certificate /etc/ssl/certs/ragwall.crt;
    ssl_certificate_key /etc/ssl/private/ragwall.key;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

---

## Scaling

### Horizontal Scaling

**Load balancer + multiple instances:**

```
                 ┌───────────────┐
                 │ Load Balancer │
                 └───────┬───────┘
                         │
        ┌────────────────┼────────────────┐
        │                │                │
        ▼                ▼                ▼
   ┌─────────┐     ┌─────────┐     ┌─────────┐
   │ Instance│     │ Instance│     │ Instance│
   │    1    │     │    2    │     │    3    │
   └─────────┘     └─────────┘     └─────────┘
```

**Auto-scaling (Kubernetes):**
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ragwall-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ragwall
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

### Vertical Scaling

**Resource allocation guidelines:**

| Mode | CPU | Memory | GPU | Throughput |
|------|-----|--------|-----|------------|
| Regex only | 2 cores | 1GB | No | 5,000 req/s |
| + Transformer (CPU) | 4 cores | 2GB | No | 50 req/s |
| + Transformer (GPU) | 4 cores | 4GB | 1x T4 | 200 req/s |
| + Transformer (GPU) | 8 cores | 8GB | 1x A100 | 500 req/s |

---

## Troubleshooting

### Issue: Slow response times

**Diagnosis:**
```bash
# Check latency
time curl -X POST http://localhost:8000/v1/sanitize \
  -H "Content-Type: application/json" \
  -d '{"query": "test"}'
```

**Solutions:**
1. Enable caching
2. Use GPU if available
3. Reduce transformer usage (increase regex coverage)
4. Add more workers

### Issue: Out of memory

**Symptoms:** Container/process killed, OOMKilled

**Solutions:**
```bash
# 1. Reduce batch size
export RAGWALL_TRANSFORMER_BATCH_SIZE=4

# 2. Use smaller model
export RAGWALL_TRANSFORMER_MODEL=distilbert-base-uncased

# 3. Increase memory limit (Kubernetes)
resources:
  limits:
    memory: "4Gi"  # Increase from 2Gi
```

---

## Deployment Checklist

### Pre-Deployment

- [ ] Model trained and validated
- [ ] Configuration tested in staging
- [ ] Health checks configured
- [ ] Monitoring enabled
- [ ] Logging configured
- [ ] Rate limiting enabled
- [ ] Authentication configured
- [ ] HTTPS/TLS enabled

### Deployment

- [ ] Deploy to production
- [ ] Run smoke tests
- [ ] Monitor metrics for 1 hour
- [ ] Check error rates
- [ ] Verify latency SLA
- [ ] Test rollback procedure

### Post-Deployment

- [ ] Set up alerts (latency, errors, resource usage)
- [ ] Document deployment process
- [ ] Train team on monitoring
- [ ] Schedule weekly reviews
- [ ] Plan for model updates

---

## Further Reading

- [Architecture Guide](ARCHITECTURE.md)
- [Fine-Tuning Guide](FINE_TUNING_GUIDE.md)
- [API Reference](API_REFERENCE.md)
- [Troubleshooting Guide](TROUBLESHOOTING.md)
