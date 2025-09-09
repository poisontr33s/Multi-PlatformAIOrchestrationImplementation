#!/bin/bash
# Development setup script for Multi-Platform AI Orchestration Implementation
# Sets up uv-based Python environment and development tools

set -e

echo "🔧 Setting up development environment..."

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ uv is not installed. Please install uv first:"
    echo "   curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment with uv..."
    uv venv
fi

# Install dependencies with dev extras
echo "📦 Installing dependencies..."
uv sync --extra dev

# Install pre-commit hooks if available
if [ -f ".pre-commit-config.yaml" ]; then
    echo "🪝 Installing pre-commit hooks..."
    uv run pre-commit install
fi

echo "✅ Development environment setup complete!"
echo ""
echo "To activate the virtual environment:"
echo "  source .venv/bin/activate"
echo ""
echo "To run commands with uv:"
echo "  uv run ruff check ."
echo "  uv run black --check ."
echo "  uv run pytest -q"
echo "  uv run uvicorn ai_orchestration.api.server:app --reload --port 8000"