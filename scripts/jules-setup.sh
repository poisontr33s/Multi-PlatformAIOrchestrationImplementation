#!/bin/bash
# Jules asynchronous agent setup script
# Installs dependencies and runs lint/tests for Jules agent development

set -e

echo "🤖 Setting up Jules asynchronous agent environment..."

# Run the main dev setup first
./scripts/dev-setup.sh

echo "🧹 Running linting checks..."
uv run ruff check . --fix
uv run black .

echo "🧪 Running tests..."
uv run pytest -q

# Check if bun is available for Node CLI bridge
if command -v bun &> /dev/null; then
    echo "✅ bun is available for Node CLI bridge"
elif command -v npx &> /dev/null; then
    echo "⚠️  bun not found, npx fallback available"
else
    echo "❌ Neither bun nor npx found. Install Node.js/npm or bun for CLI bridge functionality"
fi

echo "✅ Jules agent environment setup complete!"
echo ""
echo "Jules agent configuration:"
echo "  - Documentation: docs/JULES.md"
echo "  - File block format: See docs/JULES.md for agent emission patterns"
echo "  - Apply file blocks: python scripts/apply_file_blocks.py < plan.md"