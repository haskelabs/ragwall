# RagWall Deployment Guide

## Quick Start (5 minutes)

### Option 1: Local Development
```bash
# Install and run
pip install -r requirements.txt
RAGWALL_VECTORS=experiments/results/tiny_jb_vectors.pt python scripts/serve_api.py

# Open demo site
open index.html
```

### Option 2: Docker (Recommended)
```bash
# Build and run with Docker Compose
docker-compose up -d

# Access at:
# - API: http://localhost:8000
# - Demo: http://localhost:80
```

## Production Deployment Options

### 1. **Cheapest: DigitalOcean Droplet** ($12/mo)
Perfect for MVP and early customers.

```bash
# Create $12/mo droplet (2GB RAM, 1 CPU)
doctl compute droplet create ragwall \
  --size s-1vcpu-2gb \
  --image docker-20-04 \
  --region nyc1

# SSH and deploy
ssh root@your-droplet-ip
git clone https://github.com/yourusername/ragwall
cd ragwall
docker-compose up -d
```

**Pros:** Simple, cheap, full control  
**Cons:** Manual scaling, no built-in redundancy

### 2. **Recommended: Render.com** ($25/mo)
Best balance of simplicity and features.

1. Push code to GitHub
2. Connect GitHub repo to Render
3. Deploy automatically on push

**Pros:** Auto-SSL, auto-deploy, zero-downtime deploys  
**Cons:** Less control than raw VPS

### 3. **Scalable: AWS App Runner** ($50-200/mo)
For growing customer base.

```bash
# Deploy with AWS CLI
aws apprunner create-service \
  --service-name "ragwall-api" \
  --source-configuration '{
    "ImageRepository": {
      "ImageIdentifier": "public.ecr.aws/ragwall:latest",
      "ImageConfiguration": {
        "Port": "8000"
      }
    }
  }'
```

**Pros:** Auto-scaling, managed, AWS ecosystem  
**Cons:** More complex, AWS lock-in

### 4. **Enterprise: Kubernetes** (Variable cost)
For large deployments or on-premise requirements.

```bash
# Deploy to K8s
kubectl apply -f k8s/
kubectl get services ragwall-service
```

**Pros:** Maximum flexibility, multi-cloud, on-premise option  
**Cons:** Complex, requires K8s expertise

## Architecture Patterns

### Microservice Pattern (Recommended)
```
┌──────────┐     ┌──────────┐     ┌──────────┐
│ Customer │────▶│ RagWall  │────▶│ Customer │
│   App    │     │   API    │     │ Vector DB│
└──────────┘     └──────────┘     └──────────┘
     │                                  ▲
     └──────────────────────────────────┘
            (bypass on safe queries)
```

### SDK Pattern (Lowest Latency)
```python
# Future: Python SDK
from ragwall import RagWallClient

client = RagWallClient(api_key="...")
safe_query = client.sanitize(user_query)
embeddings = embed(safe_query)
results = vector_db.search(embeddings)
safe_results = client.rerank(results)
```

## Configuration

### Environment Variables
```bash
RAGWALL_VECTORS=path/to/vectors.pt  # Required
RAGWALL_PORT=8000                    # Optional (default: 8000)
RAGWALL_HOST=0.0.0.0                # Optional (default: 127.0.0.1)
RAGWALL_LOG_LEVEL=INFO              # Optional (default: INFO)
```

### API Authentication (Coming Soon)
```python
# Future: API key authentication
headers = {
    "Authorization": "Bearer YOUR_API_KEY",
    "Content-Type": "application/json"
}
```

## Monitoring

### Health Checks
```bash
# Basic health check
curl http://localhost:8000/health

# With monitoring service
uptimerobot.com → http://api.ragwall.ai/health
```

### Metrics (Coming Soon)
- Request count by endpoint
- P50/P95/P99 latencies
- Risk detection rate
- API key usage

## Scaling Considerations

### Current Performance
- **Single instance:** ~500 req/sec
- **P95 latency:** ≤15ms
- **Memory usage:** ~200MB
- **CPU:** 0.5-1 core

### Scaling Strategy
1. **Vertical:** Upgrade to larger instance (immediate)
2. **Horizontal:** Add load balancer + multiple instances
3. **Caching:** Redis for repeated queries
4. **CDN:** CloudFlare for static demo site

## Security Checklist

- [ ] HTTPS/TLS enabled
- [ ] API rate limiting configured
- [ ] CORS properly configured
- [ ] Secrets in environment variables
- [ ] Regular security updates
- [ ] Log aggregation setup
- [ ] Backup strategy defined

## Cost Estimates

| Stage | Users | Infrastructure | Monthly Cost |
|-------|-------|---------------|--------------|
| MVP | <100 | 1x DigitalOcean Droplet | $12 |
| Growth | 100-1k | Render.com + CDN | $25-50 |
| Scale | 1k-10k | AWS App Runner + RDS | $200-500 |
| Enterprise | 10k+ | K8s cluster + monitoring | $1000+ |

## Support

- Documentation: [docs.ragwall.ai](https://docs.ragwall.ai)
- Email: support@ragwall.ai
- Slack: [ragwall.slack.com](https://ragwall.slack.com)