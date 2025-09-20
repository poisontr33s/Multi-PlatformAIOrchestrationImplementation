#!/bin/bash
# setup-local.sh - Local development setup script
# This script provides fallbacks for local environments without uv/bun

set -e

echo "🏠 Setting up Multi-Platform AI Orchestration for local development..."

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

# Detect Python version
PYTHON_CMD="python3"
if ! check_command python3; then
    if check_command python; then
        PYTHON_CMD="python"
    else
        echo "❌ No Python found. Please install Python 3.8+"
        exit 1
    fi
fi

echo "🐍 Using Python: $($PYTHON_CMD --version)"

# Setup Python environment (prefer uv, fallback to pip)
echo "📦 Setting up Python environment..."
if check_command uv; then
    echo "Using uv for Python package management..."
    uv venv .venv --python $PYTHON_CMD
    source .venv/bin/activate
    uv pip install -e .
    uv pip install pytest pytest-asyncio pytest-cov black isort flake8 mypy
elif check_command pip || check_command pip3; then
    echo "Using traditional pip/venv setup..."
    $PYTHON_CMD -m venv .venv
    source .venv/bin/activate
    
    # Upgrade pip
    python -m pip install --upgrade pip
    
    # Install the package and dependencies
    pip install -e .
    pip install pytest pytest-asyncio pytest-cov black isort flake8 mypy
else
    echo "❌ No package manager found. Please install pip or uv."
    exit 1
fi

# Setup Node.js environment (prefer bun, fallback to npm)
echo "🟢 Setting up Node.js environment..."
if check_command bun; then
    echo "Using Bun for Node.js package management..."
    bun install
    
    echo "🔍 Testing bunx capabilities..."
    if check_command bunx; then
        echo "✅ bunx is available"
    else
        echo "⚠️  bunx not available, but bun is installed"
    fi
    
elif check_command npm; then
    echo "Using npm for Node.js package management..."
    npm install
    
    echo "🔍 Checking npx availability..."
    if check_command npx; then
        echo "✅ npx is available"
    else
        echo "⚠️  npx not available"
    fi
    
elif check_command yarn; then
    echo "Using Yarn for Node.js package management..."
    yarn install
else
    echo "⚠️  No Node.js package manager found."
    echo "📝 To enable full functionality, please install:"
    echo "   - Bun: https://bun.sh/docs/installation" 
    echo "   - Or Node.js + npm: https://nodejs.org/"
    echo ""
    echo "Continuing with Python-only setup..."
fi

# Run tests
echo "🧪 Running tests..."
python -m pytest tests/ -v

# Run linting if tools are available
echo "🔍 Running code quality checks..."
if command -v black &> /dev/null; then
    echo "Running Black formatter check..."
    black --check src/ tests/ || echo "⚠️  Black formatting issues found. Run: black src/ tests/"
fi

if command -v isort &> /dev/null; then
    echo "Running import sorting check..."
    isort --check-only src/ tests/ || echo "⚠️  Import sorting issues found. Run: isort src/ tests/"
fi

if command -v flake8 &> /dev/null; then
    echo "Running flake8 linting..."
    flake8 src/ tests/ || echo "⚠️  Linting issues found"
fi

# Test CLI
echo "🎯 Testing CLI functionality..."
python -m ai_orchestration.cli --help > /dev/null && echo "✅ Python CLI is working"

# Create activation script with detected tools
echo "📝 Creating environment activation script..."
cat > activate.sh << EOF
#!/bin/bash
# Activate the AI Orchestration environment
source .venv/bin/activate

# Add paths for various tools
export PATH="\$HOME/.cargo/bin:\$PATH"  # for uv
export PATH="\$HOME/.bun/bin:\$PATH"    # for bun

echo "🚀 AI Orchestration environment activated!"
echo "   Python: \$(python --version)"
EOF

# Add Node.js info if available
if command -v bun &> /dev/null; then
    echo 'echo "   Bun: $(bun --version)"' >> activate.sh
elif command -v node &> /dev/null; then
    echo 'echo "   Node.js: $(node --version)"' >> activate.sh
fi

# Add CLI info
cat >> activate.sh << 'EOF'
echo "   CLI: ai-orchestrator --help"
echo ""
echo "🔧 Available commands:"
echo "   ai-orchestrator [command]    # Python CLI"
EOF

if command -v bun &> /dev/null; then
    cat >> activate.sh << 'EOF'
echo "   bun run [script]             # Node.js scripts"
echo "   bunx [package]               # Run packages directly"
EOF
elif command -v npm &> /dev/null; then
    cat >> activate.sh << 'EOF'
echo "   npm run [script]             # Node.js scripts"  
echo "   npx [package]                # Run packages directly"
EOF
fi

cat >> activate.sh << 'EOF'
echo ""
EOF

chmod +x activate.sh

# Create development helper scripts
echo "📝 Creating development helper scripts..."

# Python formatter script
cat > format-python.sh << 'EOF'
#!/bin/bash
echo "🎨 Formatting Python code..."
black src/ tests/
isort src/ tests/
echo "✅ Python formatting complete!"
EOF
chmod +x format-python.sh

# Test runner script  
cat > run-tests.sh << 'EOF'
#!/bin/bash
echo "🧪 Running all tests..."
source .venv/bin/activate
python -m pytest tests/ -v --cov=ai_orchestration
echo "✅ Tests complete!"
EOF
chmod +x run-tests.sh

echo ""
echo "🎉 Local development setup complete!"
echo ""
echo "📋 Next steps:"
echo "   1. Run: source activate.sh"
echo "   2. Test: ai-orchestrator status"  
echo "   3. Initialize: ai-orchestrator init"
echo ""
echo "🛠️  Development helpers:"
echo "   ./format-python.sh           # Format Python code"
echo "   ./run-tests.sh              # Run tests with coverage"
echo ""
echo "📚 For full functionality, consider installing:"
if ! command -v uv &> /dev/null; then
    echo "   - uv (Python): curl -LsSf https://astral.sh/uv/install.sh | sh"
fi
if ! command -v bun &> /dev/null && ! command -v node &> /dev/null; then
    echo "   - Bun (Node.js): curl -fsSL https://bun.sh/install | bash"
    echo "   - Or Node.js: https://nodejs.org/"
fi
echo ""
echo "✨ Happy developing!"