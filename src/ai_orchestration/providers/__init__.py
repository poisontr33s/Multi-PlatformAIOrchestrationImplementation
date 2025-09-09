"""
Provider adapters for multi-platform AI orchestration.

This module provides unified adapters for different AI providers:
- Google GenAI (new google-genai SDK)
- Anthropic Claude
- OpenAI GPT
- Hugging Face (OSS models)
"""

from .base import (
    AIProvider, AIModel, ChatMessage, GenerationRequest, GenerationResponse, 
    ProviderManager, MessageRole, ModelType
)

# Import providers with error handling for optional dependencies
try:
    from .google_genai import GoogleGenAIProvider
except ImportError:
    GoogleGenAIProvider = None

try:
    from .anthropic import AnthropicProvider
except ImportError:
    AnthropicProvider = None

try:
    from .openai import OpenAIProvider
except ImportError:
    OpenAIProvider = None

try:
    from .huggingface import HuggingFaceProvider
except ImportError:
    HuggingFaceProvider = None

__all__ = [
    # Base classes
    "AIProvider",
    "AIModel", 
    "ChatMessage",
    "GenerationRequest",
    "GenerationResponse",
    "ProviderManager",
    "MessageRole",
    "ModelType",
    # Provider implementations (may be None if dependencies not installed)
    "GoogleGenAIProvider",
    "AnthropicProvider",
    "OpenAIProvider", 
    "HuggingFaceProvider",
]