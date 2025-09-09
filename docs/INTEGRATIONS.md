# Integration Guide - Multi-Platform AI Orchestration

## Overview

This document provides comprehensive integration guidance for developers, DevOps engineers, and system administrators who want to integrate the Multi-Platform AI Orchestration system into their existing workflows and infrastructure.

## Table of Contents

1. [Quick Integration](#quick-integration)
2. [Runtime Requirements](#runtime-requirements)  
3. [Provider Integration](#provider-integration)
4. [API Integration](#api-integration)
5. [CLI Integration](#cli-integration)
6. [Docker Integration](#docker-integration)
7. [CI/CD Integration](#cicd-integration)
8. [Monitoring & Observability](#monitoring--observability)
9. [Security Considerations](#security-considerations)
10. [Scaling & Performance](#scaling--performance)

## Quick Integration

### For Jules/Codespace Environments

```bash
# 1. Clone and setup
git clone https://github.com/poisontr33s/Multi-PlatformAIOrchestrationImplementation.git
cd Multi-PlatformAIOrchestrationImplementation

# 2. Run Jules-optimized setup
./setup-jules.sh

# 3. Configure providers (set at least one)
export OPENAI_API_KEY="your_openai_key"
export ANTHROPIC_API_KEY="your_anthropic_key"  
export GOOGLE_API_KEY="your_google_key"

# 4. Start using
source activate.sh
ai-orchestrator init
```

### For Local Development

```bash
# 1. Setup with fallbacks
./setup-local.sh

# 2. Activate environment  
source activate.sh

# 3. Verify installation
bun run setup-checks
```

## Runtime Requirements

### Minimum Requirements

- **Python**: 3.8+ (3.12+ recommended)
- **Node.js**: 18+ or **Bun**: 1.0+ (for CLI bridges)
- **Memory**: 512MB minimum, 2GB recommended
- **Storage**: 100MB for installation, 1GB for full operation

### Recommended Stack

- **Python**: 3.12 with `uv` package manager
- **Node.js Runtime**: Bun 1.1+ for optimal performance
- **Process Manager**: PM2 or Docker for production
- **Reverse Proxy**: Nginx or Traefik for API access

### Installation Methods

#### Method 1: uv (Recommended)

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Setup project
uv venv .venv
source .venv/bin/activate  
uv pip install -e .
```

#### Method 2: Traditional pip

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

#### Method 3: With Bun

```bash
# Install Bun
curl -fsSL https://bun.sh/install | bash

# Setup Node.js dependencies
bun install
```

## Provider Integration

### Supported Providers

| Provider | SDK | Models | Features |
|----------|-----|--------|----------|
| **Google Gemini** | `google-genai` | Gemini 1.5 Pro/Flash | Long context, multimodal |
| **Anthropic Claude** | `anthropic` | Claude 3.5 Sonnet/Haiku | Reasoning, analysis |
| **OpenAI GPT** | `openai` | GPT-4, GPT-3.5 | General purpose, coding |
| **Hugging Face** | `huggingface_hub` | Llama, Mistral, etc. | Open source, privacy |

### Provider Configuration

#### Environment Variables Method

```bash
# Google Gemini
export GOOGLE_API_KEY="AIza..."
# Alternative name
export GEMINI_API_KEY="AIza..."

# Anthropic Claude
export ANTHROPIC_API_KEY="sk-ant-..."
# Alternative name  
export CLAUDE_API_KEY="sk-ant-..."

# OpenAI
export OPENAI_API_KEY="sk-..."

# Hugging Face (optional)
export HUGGINGFACE_API_KEY="hf_..."
export HF_TOKEN="hf_..."
```

#### Configuration File Method

```yaml
# config/providers.yaml
providers:
  google_genai:
    api_key: "${GOOGLE_API_KEY}"
    default_model: "gemini-1.5-pro"
    
  anthropic:
    api_key: "${ANTHROPIC_API_KEY}"
    default_model: "claude-3-5-sonnet-20241022"
    
  openai:
    api_key: "${OPENAI_API_KEY}"
    default_model: "gpt-4-turbo-preview"
    
  huggingface:
    api_key: "${HF_TOKEN}"
    default_model: "meta-llama/Llama-3.2-3B-Instruct"
```

#### Programmatic Configuration

```python
from ai_orchestration.providers import ProviderManager, OpenAIProvider, AnthropicProvider

# Initialize manager
manager = ProviderManager()

# Add providers
openai_provider = OpenAIProvider(api_key="sk-...")
anthropic_provider = AnthropicProvider(api_key="sk-ant-...")

manager.register_provider(openai_provider, is_default=True)
manager.register_provider(anthropic_provider)

# Use the manager
response = await manager.generate(request)
```

## API Integration

### OpenAPI Specification

The system exposes a fully OpenAPI 3.0 compliant REST API. Access the interactive documentation at:

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`
- **OpenAPI JSON**: `http://localhost:8000/openapi.json`

### Core Endpoints

#### 1. Health Check

```bash
curl -X GET "http://localhost:8000/health"
```

Response:
```json
{
  "status": "healthy",
  "providers": {
    "openai": true,
    "anthropic": true,
    "google-genai": false
  },
  "timestamp": "2024-01-01T12:00:00Z"
}
```

#### 2. List Models

```bash
curl -X GET "http://localhost:8000/models"
```

Response:
```json
[
  {
    "id": "gpt-4-turbo-preview",
    "name": "GPT-4 Turbo",
    "provider": "openai",
    "type": "chat",
    "context_length": 128000,
    "max_output_tokens": 4096,
    "capabilities": ["chat", "reasoning"]
  }
]
```

#### 3. Generate Text

```bash
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4",
    "prompt": "Explain quantum computing",
    "max_tokens": 100,
    "temperature": 0.7
  }'
```

#### 4. Chat Completion

```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-3-5-sonnet-20241022",
    "messages": [
      {"role": "user", "content": "Hello!"}
    ],
    "max_tokens": 100
  }'
```

### Client Libraries

#### Python Client

```python
import httpx
import asyncio

class AIOrchestrationClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        
    async def chat(self, model, messages, **kwargs):
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/chat",
                json={
                    "model": model,
                    "messages": messages,
                    **kwargs
                }
            )
            return response.json()

# Usage
client = AIOrchestrationClient()
result = await client.chat(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

#### JavaScript/TypeScript Client

```typescript
class AIOrchestrationClient {
  constructor(private baseUrl = 'http://localhost:8000') {}
  
  async chat(model: string, messages: any[], options: any = {}) {
    const response = await fetch(`${this.baseUrl}/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model,
        messages,
        ...options
      })
    });
    
    return response.json();
  }
}

// Usage
const client = new AIOrchestrationClient();
const result = await client.chat('claude-3-5-sonnet-20241022', [
  { role: 'user', content: 'Hello!' }
]);
```

## CLI Integration

### Python CLI

The primary CLI provides full orchestration capabilities:

```bash
# Initialize system
ai-orchestrator init --subscription --emergent

# Submit tasks
ai-orchestrator submit-task task_001 \
  --type autonomous_strategic_implementation \
  --priority critical_infrastructure \
  --payload '{"objective": "optimization"}'

# Monitor status
ai-orchestrator status

# View patterns
ai-orchestrator patterns
```

### Node.js CLI Bridges

Lightweight bridges for direct provider access:

```bash
# Gemini CLI
bun run gemini --prompt "Explain AI" --model gemini-1.5-pro

# Claude CLI  
bun run claude --prompt "Review code" --file code.py

# OpenAI CLI
bun run openai --prompt "Generate docs" --temperature 0.3
```

### Shell Integration

Add to your `.bashrc` or `.zshrc`:

```bash
# AI Orchestration aliases
alias aic='ai-orchestrator'
alias gemini='bun run gemini'
alias claude='bun run claude'  
alias gpt='bun run openai'

# Quick functions
ask_ai() {
  local provider=${1:-openai}
  local prompt="$2"
  bun run $provider --prompt "$prompt"
}

# Usage: ask_ai claude "Explain this code"
```

## Docker Integration

### Docker Compose Setup

```yaml
# docker-compose.yml
version: '3.8'

services:
  ai-orchestration:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - GOOGLE_API_KEY=${GOOGLE_API_KEY}
    volumes:
      - ./config:/app/config
      - ./logs:/app/logs
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - ai-orchestration
```

### Dockerfile

```dockerfile
FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:$PATH"

# Install Bun
RUN curl -fsSL https://bun.sh/install | bash
ENV PATH="/root/.bun/bin:$PATH"

WORKDIR /app

# Copy dependency files
COPY pyproject.toml package.json ./

# Install dependencies
RUN uv venv .venv && \
    . .venv/bin/activate && \
    uv pip install -e .

RUN bun install

# Copy application
COPY . .

# Activate Python environment
ENV PATH="/app/.venv/bin:$PATH"

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Start server
CMD ["python", "-m", "uvicorn", "ai_orchestration.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

## CI/CD Integration

### GitHub Actions

```yaml
# .github/workflows/ci.yml
name: CI/CD Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Setup Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.12'
        
    - name: Setup Bun
      uses: oven-sh/setup-bun@v1
      with:
        bun-version: latest
        
    - name: Install uv
      run: curl -LsSf https://astral.sh/uv/install.sh | sh
      
    - name: Install dependencies
      run: |
        source ~/.cargo/env
        uv venv .venv
        source .venv/bin/activate
        uv pip install -e .
        bun install
        
    - name: Run tests
      run: |
        source .venv/bin/activate
        pytest tests/ -v --cov=ai_orchestration
        
    - name: Run linting
      run: |
        source .venv/bin/activate
        black --check src/ tests/
        isort --check-only src/ tests/
        flake8 src/ tests/
        
  deploy:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Deploy to production
      run: |
        # Add your deployment steps here
        echo "Deploying to production..."
```

### GitLab CI

```yaml
# .gitlab-ci.yml
stages:
  - test
  - build
  - deploy

variables:
  PYTHON_VERSION: "3.12"

before_script:
  - curl -LsSf https://astral.sh/uv/install.sh | sh
  - source ~/.cargo/env
  - curl -fsSL https://bun.sh/install | bash
  - source ~/.bashrc

test:
  stage: test
  script:
    - uv venv .venv
    - source .venv/bin/activate
    - uv pip install -e .
    - bun install
    - pytest tests/ -v
    - black --check src/ tests/
    
build:
  stage: build
  script:
    - docker build -t ai-orchestration:$CI_COMMIT_SHA .
    
deploy:
  stage: deploy
  only:
    - main
  script:
    - kubectl apply -f k8s/
```

## Monitoring & Observability

### Metrics Collection

The system exposes Prometheus-compatible metrics:

```python
# Custom metrics integration
from prometheus_client import Counter, Histogram, Gauge

request_count = Counter('ai_requests_total', 'Total AI requests', ['provider', 'model'])
request_duration = Histogram('ai_request_duration_seconds', 'Request duration')
active_providers = Gauge('ai_providers_active', 'Number of active providers')
```

### Logging Configuration

```python
# logging_config.py
import structlog

structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)
```

### Health Monitoring

```bash
#!/bin/bash
# health_check.sh

# Check API health
HEALTH=$(curl -s http://localhost:8000/health | jq -r '.status')

if [ "$HEALTH" != "healthy" ]; then
    echo "ALERT: AI Orchestration system is unhealthy"
    # Send alert to monitoring system
    exit 1
fi

# Check provider health
PROVIDERS=$(curl -s http://localhost:8000/health | jq -r '.providers | to_entries[] | select(.value == false) | .key')

if [ -n "$PROVIDERS" ]; then
    echo "WARNING: Some providers are down: $PROVIDERS"
fi

echo "System healthy"
```

## Security Considerations

### API Security

1. **Authentication**: Implement JWT or API key authentication
2. **Rate Limiting**: Use rate limiting to prevent abuse
3. **CORS**: Configure CORS policies appropriately
4. **HTTPS**: Always use HTTPS in production

```python
# security_config.py
from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    # Implement your token verification logic
    if not verify_jwt_token(credentials.credentials):
        raise HTTPException(status_code=401, detail="Invalid token")
    return credentials.credentials

# Protect endpoints
@app.post("/chat", dependencies=[Depends(verify_token)])
async def chat_endpoint(request: ChatRequest):
    # Your implementation
    pass
```

### API Key Management

1. **Environment Variables**: Never commit API keys to code
2. **Secret Management**: Use tools like HashiCorp Vault, AWS Secrets Manager
3. **Rotation**: Regularly rotate API keys
4. **Least Privilege**: Use keys with minimal required permissions

```bash
# Using AWS Secrets Manager
aws secretsmanager get-secret-value \
  --secret-id ai-orchestration/api-keys \
  --query SecretString --output text | jq -r '.OPENAI_API_KEY'
```

### Network Security

```nginx
# nginx.conf - Security headers
server {
    listen 443 ssl http2;
    server_name your-domain.com;
    
    # Security headers
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains";
    
    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    limit_req zone=api burst=20 nodelay;
    
    location / {
        proxy_pass http://ai-orchestration:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

## Scaling & Performance

### Horizontal Scaling

```yaml
# kubernetes/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ai-orchestration
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ai-orchestration
  template:
    metadata:
      labels:
        app: ai-orchestration
    spec:
      containers:
      - name: ai-orchestration
        image: ai-orchestration:latest
        ports:
        - containerPort: 8000
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: ai-keys
              key: openai-key
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
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
          initialDelaySeconds: 5
          periodSeconds: 5
```

### Load Balancing

```python
# load_balancer.py
import random
from typing import List
from ai_orchestration.providers import ProviderManager

class LoadBalancingProviderManager(ProviderManager):
    def __init__(self, strategy='round_robin'):
        super().__init__()
        self.strategy = strategy
        self._counter = 0
        
    async def generate(self, request, provider_name=None):
        if provider_name:
            return await super().generate(request, provider_name)
            
        # Load balancing logic
        available_providers = [
            name for name, provider in self._providers.items()
            if await provider.health_check()
        ]
        
        if not available_providers:
            raise RuntimeError("No healthy providers available")
            
        if self.strategy == 'round_robin':
            provider = available_providers[self._counter % len(available_providers)]
            self._counter += 1
        elif self.strategy == 'random':
            provider = random.choice(available_providers)
        elif self.strategy == 'least_loaded':
            # Implement least loaded logic
            provider = self._get_least_loaded_provider(available_providers)
            
        return await super().generate(request, provider)
```

### Caching

```python
# caching.py
import redis
import json
import hashlib
from functools import wraps

redis_client = redis.Redis(host='localhost', port=6379, db=0)

def cache_response(ttl=3600):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Create cache key
            cache_key = hashlib.md5(
                json.dumps({'args': args, 'kwargs': kwargs}, sort_keys=True).encode()
            ).hexdigest()
            
            # Try to get from cache
            cached = redis_client.get(cache_key)
            if cached:
                return json.loads(cached)
                
            # Execute function
            result = await func(*args, **kwargs)
            
            # Cache result
            redis_client.setex(cache_key, ttl, json.dumps(result))
            
            return result
        return wrapper
    return decorator

# Usage
@cache_response(ttl=1800)  # 30 minutes
async def cached_generate(request):
    return await provider.generate(request)
```

## Troubleshooting

### Common Issues

1. **Provider Initialization Fails**
   ```bash
   # Check API keys
   env | grep -E "(GOOGLE|ANTHROPIC|OPENAI).*API"
   
   # Test connectivity
   curl -H "Authorization: Bearer $OPENAI_API_KEY" https://api.openai.com/v1/models
   ```

2. **Port Already in Use**
   ```bash
   # Find process using port 8000
   lsof -i :8000
   
   # Kill process
   kill -9 <PID>
   
   # Or use different port
   uvicorn ai_orchestration.api:app --port 8001
   ```

3. **Import Errors**
   ```bash
   # Check Python path
   python -c "import sys; print(sys.path)"
   
   # Reinstall in development mode
   pip install -e .
   ```

### Performance Issues

1. **Slow Response Times**
   - Check provider health: `curl http://localhost:8000/health`
   - Monitor network latency to provider APIs
   - Implement caching for repeated requests
   - Use streaming for long responses

2. **High Memory Usage**
   - Monitor process memory: `ps aux | grep python`
   - Implement request queuing
   - Scale horizontally instead of vertically

3. **Provider Rate Limits**
   - Implement exponential backoff
   - Distribute load across multiple providers
   - Cache responses when possible

### Debugging

```bash
# Enable debug logging
export AI_ORCHESTRATION_LOG_LEVEL=DEBUG

# Trace HTTP requests
export HTTPX_LOG_LEVEL=DEBUG

# Run with profiling
python -m cProfile -o profile.stats -m uvicorn ai_orchestration.api:app

# Analyze profile
python -c "import pstats; pstats.Stats('profile.stats').sort_stats('cumtime').print_stats(20)"
```

## Support & Resources

### Documentation
- **API Docs**: `/docs` endpoint (Swagger UI)
- **Code Examples**: `examples/` directory
- **Configuration Reference**: `config/default.yaml`

### Community
- **Issues**: GitHub Issues for bug reports
- **Discussions**: GitHub Discussions for questions
- **Contributing**: See `CONTRIBUTING.md`

### Professional Support
For enterprise deployments, consider:
- Custom provider implementations
- Performance optimization consulting  
- Security audits and compliance
- 24/7 monitoring and support

---

This integration guide covers the essential aspects of deploying and integrating the Multi-Platform AI Orchestration system. For specific use cases or advanced configurations, refer to the individual component documentation or reach out to the development team.