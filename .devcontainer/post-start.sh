#!/bin/bash

# Post-start script for Multi-Platform AI Orchestration Environment
set -e

echo "🔄 Starting post-start initialization..."

# Activate virtual environment if it exists
if [ -d "/workspace/venv" ]; then
    source /workspace/venv/bin/activate
fi

# Check GPU availability
echo "🎮 Checking GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits
else
    echo "⚠️  NVIDIA GPU not detected - some features may be limited"
fi

# Start background services if configured
if [ -f "/workspace/docker-compose.yml" ]; then
    echo "🐳 Starting background services..."
    cd /workspace && docker-compose up -d
fi

# Initialize monitoring if enabled
if [ "$AI_ORCHESTRATION_ENV" = "development" ] && [ -f "/workspace/scripts/start_monitoring.sh" ]; then
    echo "📊 Starting monitoring services..."
    bash /workspace/scripts/start_monitoring.sh
fi

# Check authentication configuration
if [ -f "/workspace/configs/auth_config.yaml" ]; then
    echo "✅ Authentication configuration found"
else
    echo "⚠️  Authentication configuration not found - please configure credentials"
fi

# Display environment status
echo "🌟 Environment Status:"
echo "   Python: $(python --version)"
echo "   Node.js: $(node --version)"
echo "   Git: $(git --version)"
echo "   Environment: $AI_ORCHESTRATION_ENV"

echo "✅ Post-start initialization completed!"