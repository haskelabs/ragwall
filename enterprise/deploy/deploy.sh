#!/bin/bash
# RagWall Deployment Script
# Supports: Local, Docker, AWS, DigitalOcean

set -e

DEPLOY_TYPE=${1:-docker}
DOMAIN=${2:-api.ragwall.ai}

echo "🚀 RagWall Deployment Script"
echo "============================="

case $DEPLOY_TYPE in
  local)
    echo "📍 Deploying locally..."
    pip install -r requirements.txt
    RAGWALL_VECTORS=experiments/results/tiny_jb_vectors.pt python scripts/serve_api.py &
    echo "✅ API running at http://localhost:8000"
    open index.html
    ;;
    
  docker)
    echo "🐳 Building Docker image..."
    cat > Dockerfile <<EOF
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
ENV RAGWALL_VECTORS=/app/experiments/results/tiny_jb_vectors.pt
CMD ["python", "scripts/serve_api.py"]
EOF
    
    docker build -t ragwall:latest .
    docker run -d -p 8000:8000 --name ragwall-api ragwall:latest
    echo "✅ Docker container running at http://localhost:8000"
    ;;
    
  aws)
    echo "☁️  Deploying to AWS App Runner..."
    
    # Create apprunner.yaml
    cat > apprunner.yaml <<EOF
version: 1.0
runtime: python311
build:
  commands:
    build:
      - pip install -r requirements.txt
run:
  runtime-version: latest
  command: python scripts/serve_api.py
  network:
    port: 8000
    env: PORT
  env:
    - name: RAGWALL_VECTORS
      value: experiments/results/tiny_jb_vectors.pt
EOF
    
    # Push to ECR
    aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin $ECR_URI
    docker tag ragwall:latest $ECR_URI/ragwall:latest
    docker push $ECR_URI/ragwall:latest
    
    # Deploy static site to S3
    aws s3 cp index.html s3://ragwall-demo/index.html --acl public-read
    aws s3 website s3://ragwall-demo --index-document index.html
    
    echo "✅ Deployed to AWS"
    echo "   API: https://${DOMAIN}"
    echo "   Demo: https://ragwall-demo.s3-website-us-east-1.amazonaws.com"
    ;;
    
  digitalocean)
    echo "💧 Deploying to DigitalOcean..."
    
    # Create docker-compose.yml
    cat > docker-compose.yml <<EOF
version: '3.8'
services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - RAGWALL_VECTORS=/app/experiments/results/tiny_jb_vectors.pt
    restart: unless-stopped
  
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./index.html:/usr/share/nginx/html/index.html
    depends_on:
      - api
    restart: unless-stopped
EOF

    # Create nginx.conf
    cat > nginx.conf <<EOF
events {
    worker_connections 1024;
}

http {
    server {
        listen 80;
        server_name ${DOMAIN};
        
        location / {
            root /usr/share/nginx/html;
            try_files \$uri \$uri/ =404;
        }
        
        location /v1/ {
            proxy_pass http://api:8000;
            proxy_set_header Host \$host;
            proxy_set_header X-Real-IP \$remote_addr;
        }
    }
}
EOF
    
    # Deploy with doctl
    doctl apps create --spec app.yaml
    echo "✅ Deployed to DigitalOcean"
    ;;
    
  render)
    echo "🎨 Deploying to Render.com..."
    
    # Create render.yaml
    cat > render.yaml <<EOF
services:
  - type: web
    name: ragwall-api
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: python scripts/serve_api.py
    envVars:
      - key: RAGWALL_VECTORS
        value: experiments/results/tiny_jb_vectors.pt
    
  - type: web
    name: ragwall-demo
    env: static
    staticPublishPath: .
    routes:
      - type: rewrite
        source: /*
        destination: /index.html
EOF
    
    echo "✅ Push to GitHub and connect to Render.com"
    ;;
    
  *)
    echo "Usage: ./deploy.sh [local|docker|aws|digitalocean|render] [domain]"
    echo ""
    echo "Examples:"
    echo "  ./deploy.sh local                    # Run locally"
    echo "  ./deploy.sh docker                   # Run in Docker"
    echo "  ./deploy.sh aws api.ragwall.ai      # Deploy to AWS"
    echo "  ./deploy.sh digitalocean ragwall.ai # Deploy to DigitalOcean"
    exit 1
    ;;
esac

echo ""
echo "📊 Next Steps:"
echo "1. Test the API: curl http://localhost:8000/health"
echo "2. Monitor logs: docker logs -f ragwall-api"
echo "3. Set up monitoring: DataDog, CloudWatch, or Prometheus"
echo "4. Configure rate limiting and API keys"
echo "5. Set up SSL certificates (Let's Encrypt)"