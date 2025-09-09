#!/bin/bash

# Post-create script for Multi-Platform AI Orchestration Environment
set -e

echo "🚀 Starting post-create setup for AI Orchestration Environment..."

# Create necessary directories
mkdir -p /workspace/.cache
mkdir -p /workspace/logs
mkdir -p /workspace/data
mkdir -p /workspace/models
mkdir -p /workspace/configs

# Set up Python environment
echo "🐍 Setting up Python environment..."
pip install --upgrade pip setuptools wheel

# Install core dependencies
pip install \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 \
    transformers \
    accelerate \
    bitsandbytes \
    sentencepiece \
    protobuf \
    numpy \
    pandas \
    scikit-learn \
    matplotlib \
    seaborn \
    jupyter \
    jupyterlab \
    fastapi \
    uvicorn \
    pydantic \
    sqlalchemy \
    alembic \
    redis \
    celery \
    pytest \
    pytest-asyncio \
    black \
    flake8 \
    pylint \
    mypy \
    pre-commit \
    python-dotenv \
    aiohttp \
    aiofiles \
    asyncio-mqtt \
    websockets \
    prometheus-client \
    structlog \
    rich \
    typer \
    click

# Install Google AI dependencies
pip install \
    google-cloud-aiplatform \
    google-cloud-storage \
    google-cloud-firestore \
    google-generativeai \
    firebase-admin

# Install Microsoft AI dependencies
pip install \
    azure-ai-ml \
    azure-cognitiveservices-language-textanalytics \
    azure-storage-blob \
    openai \
    semantic-kernel

# Install monitoring and observability
pip install \
    grafana-client \
    prometheus-client \
    jaeger-client \
    opentelemetry-api \
    opentelemetry-sdk \
    opentelemetry-instrumentation

# Set up Node.js environment
echo "📦 Setting up Node.js environment..."
npm install -g \
    typescript \
    ts-node \
    @types/node \
    firebase-tools \
    @google-cloud/functions-framework \
    prettier \
    eslint \
    @typescript-eslint/parser \
    @typescript-eslint/eslint-plugin

# Initialize project structure
echo "📁 Initializing project structure..."
if [ ! -f /workspace/package.json ]; then
    cd /workspace && npm init -y
fi

# Set up Git hooks
echo "🔧 Setting up Git hooks..."
if [ -d /workspace/.git ]; then
    cd /workspace && pre-commit install
fi

# Set up authentication placeholders
echo "🔐 Setting up authentication configuration..."
cat > /workspace/configs/auth_config_template.yaml << EOF
# Authentication Configuration Template
# Copy to auth_config.yaml and fill in your credentials

github:
  token: "YOUR_GITHUB_TOKEN"
  app_id: "YOUR_GITHUB_APP_ID"
  private_key_path: "path/to/private/key"

google:
  project_id: "YOUR_GOOGLE_PROJECT_ID"
  service_account_path: "path/to/service/account.json"
  ai_api_key: "YOUR_GOOGLE_AI_API_KEY"

microsoft:
  tenant_id: "YOUR_AZURE_TENANT_ID"
  client_id: "YOUR_AZURE_CLIENT_ID"
  client_secret: "YOUR_AZURE_CLIENT_SECRET"
  openai_api_key: "YOUR_OPENAI_API_KEY"

firebase:
  project_id: "YOUR_FIREBASE_PROJECT_ID"
  service_account_path: "path/to/firebase/service/account.json"
  api_key: "YOUR_FIREBASE_API_KEY"

jules:
  api_endpoint: "YOUR_JULES_API_ENDPOINT"
  api_key: "YOUR_JULES_API_KEY"
  webhook_secret: "YOUR_JULES_WEBHOOK_SECRET"
EOF

# Create environment file template
cat > /workspace/.env.template << EOF
# Environment Variables Template
# Copy to .env and fill in your values

# GitHub Configuration
GITHUB_TOKEN=your_github_token_here
GITHUB_APP_ID=your_github_app_id_here
GITHUB_WEBHOOK_SECRET=your_github_webhook_secret_here

# Google AI Configuration
GOOGLE_PROJECT_ID=your_google_project_id_here
GOOGLE_APPLICATION_CREDENTIALS=path/to/service_account.json
GOOGLE_AI_API_KEY=your_google_ai_api_key_here

# Microsoft AI Configuration
AZURE_TENANT_ID=your_azure_tenant_id_here
AZURE_CLIENT_ID=your_azure_client_id_here
AZURE_CLIENT_SECRET=your_azure_client_secret_here
OPENAI_API_KEY=your_openai_api_key_here

# Firebase Configuration
FIREBASE_PROJECT_ID=your_firebase_project_id_here
FIREBASE_API_KEY=your_firebase_api_key_here

# Jules Agent Configuration
JULES_API_ENDPOINT=your_jules_api_endpoint_here
JULES_API_KEY=your_jules_api_key_here

# Development Configuration
AI_ORCHESTRATION_ENV=development
DEBUG=true
LOG_LEVEL=INFO

# Resource Configuration
MAX_GPU_MEMORY=30gb
MAX_CPU_CORES=8
DEFAULT_MODEL_CACHE_SIZE=10gb
EOF

# Create copilot instructions template
mkdir -p /workspace/.github
cat > /workspace/.github/copilot-instructions.md << EOF
# GitHub Copilot Instructions for Multi-Platform AI Orchestration

## Project Overview
This is a sophisticated multi-platform AI orchestration system that coordinates between GitHub Copilot Pro+, Google AI Pro/Ultra, Microsoft AI Pro, Jules Asynchronous Coding Agent, Firebase Studio, and local Gemma 3/GPT-OSS models.

## Code Style Guidelines
- Use type hints for all Python functions
- Implement comprehensive error handling with structured logging
- Follow async/await patterns for all I/O operations
- Use dependency injection for all external service integrations
- Implement circuit breaker patterns for API calls
- Include comprehensive docstrings with examples

## Architecture Principles
- Modular design with clear separation of concerns
- Event-driven architecture with message queues
- Graceful degradation and fallback mechanisms
- Resource optimization and GPU memory management
- Security-first design with proper authentication
- Observability through metrics, logs, and traces

## Integration Patterns
- OAuth 2.1 + PKCE for all external authentications
- Rate limiting and retry logic for API calls
- Webhook handling with signature verification
- Container orchestration with health checks
- Configuration management through environment variables
- Secrets management through secure key stores

## Testing Requirements
- Unit tests for all core functionality
- Integration tests for all external service connections
- Performance tests for resource utilization
- Security tests for authentication and authorization
- End-to-end tests for complete workflows

## Documentation Standards
- API documentation with OpenAPI/Swagger
- Architecture decision records (ADRs)
- Deployment and configuration guides
- Troubleshooting and monitoring guides
- Performance optimization guides
EOF

echo "✅ Post-create setup completed successfully!"
echo "📝 Next steps:"
echo "   1. Copy .env.template to .env and configure your credentials"
echo "   2. Copy configs/auth_config_template.yaml to configs/auth_config.yaml"
echo "   3. Run 'make setup' to initialize the development environment"
echo "   4. Review .github/copilot-instructions.md for development guidelines"