FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose the API port
EXPOSE 8000

# Set environment variables
ENV RAGWALL_VECTORS=/app/experiments/results/tiny_jb_vectors.pt
ENV PYTHONUNBUFFERED=1

# Health check (use stdlib to avoid extra deps)
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD python - <<'PY' || exit 1
import urllib.request
try:
    with urllib.request.urlopen('http://localhost:8000/health', timeout=2) as r:
        exit(0 if r.status == 200 else 1)
except Exception:
    exit(1)
PY

# Run the API server
CMD ["python", "scripts/serve_api.py"]
