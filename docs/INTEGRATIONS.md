# Google Gemini Integration Guide (2025 Standards)

This guide covers the Google Gemini integration for the Multi-Platform AI Orchestration Implementation - Phase 1, updated for September 2025 with the latest Gemini 2.0+ models and enhanced capabilities.

## Setup (2025 Enhanced)

### 1. Environment Setup

Use the provided setup scripts for quick initialization with modern tooling:

```bash
# Set up development environment with uv (2025 optimized)
./scripts/dev-setup.sh

# Set up Jules agent environment (includes linting and testing)
./scripts/jules-setup.sh
```

### 2. API Key Configuration (2025 Models)

Get your Google AI Studio API key from [Google AI Studio](https://makersuite.google.com/app/apikey).

```bash
# Copy environment template
cp .env.example .env

# Edit .env and set your API key with 2025 recommended model
GOOGLE_API_KEY=your_google_api_key_here
GOOGLE_GENAI_MODEL=gemini-2.0-flash-exp  # 2025 recommended default
```

**2025 Model Recommendations:**
- **`gemini-2.0-flash-exp`**: Fastest with experimental features, 1M+ token context
- **`gemini-2.0-flash-thinking-exp`**: Advanced reasoning capabilities  
- **`gemini-2.0-flash`**: Stable production model with 1M+ tokens
- **`gemini-2.5-flash-preview`**: Latest preview features (experimental)

### 3. Install Dependencies

```bash
# Install with uv (recommended for 2025)
uv sync --extra dev

# Or with pip
pip install -e ".[dev]"
```

## Running the Server

### Start the FastAPI Server

```bash
# Using uv (recommended)
uv run uvicorn ai_orchestration.api.server:app --reload --port 8000

# Or activate venv and run directly
source .venv/bin/activate
uvicorn ai_orchestration.api.server:app --reload --port 8000
```

The server will be available at:
- API: http://localhost:8000
- Interactive Docs: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### API Endpoints

#### Health Check
```bash
curl http://localhost:8000/health
```

#### List Models
```bash
curl http://localhost:8000/models
```

#### Generate Text
```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Write a haiku about AI", "temperature": 0.7}'
```

## Available Models

The Google Gemini provider supports current Gemini models:

- **gemini-1.5-flash** - Fast and efficient for most tasks (default)
- **gemini-1.5-pro** - Highest quality for complex tasks
- **gemini-1.0-pro** - Stable and reliable for production

⚠️ **Avoid deprecated models** like "text-bison-001" or "chat-bison-001".

## Configuration Options

### Environment Variables

```bash
# Required
GOOGLE_API_KEY=your_api_key_here

# Model selection (optional, defaults to gemini-1.5-flash)
GOOGLE_GENAI_MODEL=gemini-1.5-pro

# API server configuration
API_HOST=127.0.0.1
API_PORT=8000
API_RELOAD=true

# Logging
AI_ORCHESTRATION_LOG_LEVEL=INFO
```

### Generation Parameters

When calling the `/generate` endpoint, you can specify:

- **prompt** (required): Text prompt for generation
- **max_tokens** (optional): Maximum tokens to generate (1-8192)
- **temperature** (optional): Sampling temperature (0.0-2.0)

Example:
```json
{
  "prompt": "Explain quantum computing in simple terms",
  "max_tokens": 200,
  "temperature": 0.3
}
```

## Development

### Running Tests

```bash
# Run all tests
uv run pytest -q

# Run specific test files
uv run pytest tests/test_google_imports.py -v
uv run pytest tests/test_api_endpoints.py -v

# Run with coverage
uv run pytest --cov=ai_orchestration
```

### Code Quality

```bash
# Run linting
uv run ruff check .

# Format code
uv run black .

# Fix auto-fixable issues
uv run ruff check --fix .
```

### Node CLI Bridge

The system includes a Node.js CLI bridge for future integration:

```python
from ai_orchestration.bridge.node_cli import NodeCLIBridge

bridge = NodeCLIBridge()
# Future: gemini CLI commands will be available here
```

## Troubleshooting

### Common Issues

#### 1. "Google API key is required"
```
Error: Google API key is required. Set GOOGLE_API_KEY environment variable
```
**Solution**: Set the `GOOGLE_API_KEY` environment variable with your API key from Google AI Studio.

#### 2. "Provider not initialized" (503 error)
```
{"detail":"Google Gemini provider not initialized"}
```
**Solution**: This occurs when the API key is missing or invalid. Check your `.env` file and ensure the API key is correct.

#### 3. "No module named 'google.generativeai'"
```
ModuleNotFoundError: No module named 'google.generativeai'
```
**Solution**: Install dependencies with `uv sync --extra dev` or `pip install google-generativeai`.

#### 4. Model not found errors
**Solution**: Use current model names like `gemini-1.5-flash` instead of deprecated models.

### Debug Mode

Enable debug logging for troubleshooting:

```bash
export AI_ORCHESTRATION_LOG_LEVEL=DEBUG
uv run uvicorn ai_orchestration.api.server:app --log-level debug
```

## Integration Examples

### Python Client Example

```python
import httpx

async def example_generate():
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/generate",
            json={
                "prompt": "Write a Python function to calculate fibonacci numbers",
                "max_tokens": 500,
                "temperature": 0.2
            }
        )
        return response.json()
```

### Curl Examples

```bash
# Health check
curl http://localhost:8000/health

# List available models
curl http://localhost:8000/models

# Generate text with specific parameters
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Create a REST API endpoint for user management",
    "max_tokens": 300,
    "temperature": 0.5
  }'
```

## Future Enhancements (TODOs)

The current implementation is Phase 1 (Google only). Future phases will include:

- [ ] **Vertex AI Support**: Integration with Google Cloud Vertex AI
- [ ] **Streaming Support**: Real-time text generation
- [ ] **Function Calling**: Gemini function calling capabilities
- [ ] **Multi-Provider Routing**: Anthropic Claude and OpenAI integration
- [ ] **Authentication**: API key management and user authentication
- [ ] **Rate Limiting**: Request throttling and quota management
- [ ] **Async Processing**: Background task processing
- [ ] **Caching**: Response caching for improved performance

## API Reference

See the automatically generated API documentation:
- Interactive Docs: http://localhost:8000/docs
- OpenAPI Spec: `/openapi/openapi.yaml`
- ReDoc: http://localhost:8000/redoc