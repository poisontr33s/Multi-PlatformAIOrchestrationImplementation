"""
FastAPI REST API for Multi-Platform AI Orchestration.

Provides OpenAPI-compliant endpoints for:
- /models - List available models from all providers
- /generate - Generate text/completions
- /chat - Chat completions
- /health - Health check
"""

from .main import app

__all__ = ["app"]