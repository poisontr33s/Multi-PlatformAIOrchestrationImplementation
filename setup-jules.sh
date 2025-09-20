#!/bin/bash
# setup-jules.sh - Setup script optimized for Jules VMs
# This script uses uv and validates bunx for the dual-runtime toolchain

set -e

echo "🚀 Setting up Multi-Platform AI Orchestration for Jules VM..."

# Check if we're in a Jules environment
if [[ -n "$JULES_ENV" || -n "$CODESPACE_NAME" || -f "/.dockerenv" ]]; then
    echo "✅ Jules environment detected"
    JULES_MODE=true
else
    echo "ℹ️  Running in standard environment"
    JULES_MODE=false
fi

# Function to check command availability
check_command() {
    if command -v "$1" &> /dev/null; then
        echo "✅ $1 is available"
        return 0
    else
        echo "❌ $1 is not available"
        return 1
    fi
}

# Install or check uv (Python package manager)
echo "📦 Setting up Python environment with uv..."
if ! check_command uv; then
    echo "Installing uv..."
    if [[ "$JULES_MODE" == "true" ]]; then
        # Use curl method for Jules/containerized environments
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.cargo/bin:$PATH"
    else
        # Try package manager first for local environments
        if command -v brew &> /dev/null; then
            brew install uv
        elif command -v apt-get &> /dev/null; then
            curl -LsSf https://astral.sh/uv/install.sh | sh
            export PATH="$HOME/.cargo/bin:$PATH"
        else
            curl -LsSf https://astral.sh/uv/install.sh | sh
            export PATH="$HOME/.cargo/bin:$PATH"
        fi
    fi
fi

# Create Python virtual environment with uv
echo "🐍 Creating Python virtual environment..."
uv venv .venv
source .venv/bin/activate

# Install Python dependencies
echo "📥 Installing Python dependencies with uv..."
uv pip install -e .

# Install development dependencies
echo "🛠️  Installing development dependencies..."
uv pip install pytest pytest-asyncio pytest-cov black isort flake8 mypy

# Check Bun availability
echo "🥖 Checking Bun availability..."
if ! check_command bun; then
    echo "⚠️  Bun not found. Attempting to install..."
    if [[ "$JULES_MODE" == "true" ]]; then
        # For Jules environment, try to install Bun
        curl -fsSL https://bun.sh/install | bash
        export PATH="$HOME/.bun/bin:$PATH"
    else
        echo "📝 Please install Bun manually: https://bun.sh/docs/installation"
        echo "   curl -fsSL https://bun.sh/install | bash"
    fi
fi

# Validate bunx if Bun is available
if check_command bun; then
    echo "🔍 Validating bunx capabilities..."
    if check_command bunx; then
        echo "✅ bunx is available"
        
        # Test bunx with a simple command
        if bunx --version &> /dev/null; then
            echo "✅ bunx is working correctly"
        else
            echo "⚠️  bunx may have issues, but continuing..."
        fi
    else
        echo "⚠️  bunx not found, but bun is available"
    fi
    
    # Install Node.js dependencies
    echo "📦 Installing Node.js dependencies..."
    bun install
else
    echo "⚠️  Skipping Node.js setup - Bun not available"
fi

# Run Python tests
echo "🧪 Running Python tests..."
python -m pytest tests/ -v

# Run Python linting
echo "🔍 Running Python linting..."
if command -v black &> /dev/null; then
    black --check src/ tests/ || echo "⚠️  Black formatting issues found"
fi

if command -v isort &> /dev/null; then
    isort --check-only src/ tests/ || echo "⚠️  Import sorting issues found"
fi

if command -v flake8 &> /dev/null; then
    flake8 src/ tests/ || echo "⚠️  Flake8 issues found"
fi

# Test CLI
echo "🎯 Testing AI Orchestration CLI..."
python -m ai_orchestration.cli --help > /dev/null && echo "✅ CLI is working"

# Create environment activation script
echo "📝 Creating environment activation script..."
cat > activate.sh << 'EOF'
#!/bin/bash
# Activate the AI Orchestration environment
source .venv/bin/activate
export PATH="$HOME/.cargo/bin:$PATH"
export PATH="$HOME/.bun/bin:$PATH"
echo "🚀 AI Orchestration environment activated!"
echo "   Python: $(python --version)"
if command -v bun &> /dev/null; then
    echo "   Bun: $(bun --version)"
fi
echo "   CLI: ai-orchestrator --help"
EOF
chmod +x activate.sh

echo ""
echo "🎉 Setup complete for Jules VM!"
echo ""
echo "📋 Next steps:"
echo "   1. Run: source activate.sh"
echo "   2. Test: ai-orchestrator status"
echo "   3. Initialize: ai-orchestrator init"
echo ""
echo "🔧 Available commands:"
echo "   Python CLI: ai-orchestrator [command]"
if command -v bun &> /dev/null; then
    echo "   Node CLI: bun run gemini | claude | openai"
    echo "   Dev tools: bun run dev | test | lint"
fi
echo ""
echo "✨ Happy orchestrating!"