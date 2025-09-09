"""
Provider manager dependency for FastAPI.
"""

import os
from typing import Optional
from functools import lru_cache

from ..providers import (
    ProviderManager,
    GoogleGenAIProvider,
    AnthropicProvider, 
    OpenAIProvider,
    HuggingFaceProvider
)


@lru_cache()
def get_provider_manager() -> ProviderManager:
    """
    Get or create the global provider manager instance.
    
    This function initializes all available AI providers based on
    environment variables and registers them with the manager.
    """
    manager = ProviderManager()
    
    # Initialize Google GenAI provider
    if GoogleGenAIProvider is not None:
        google_api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if google_api_key:
            try:
                google_provider = GoogleGenAIProvider(api_key=google_api_key)
                manager.register_provider(google_provider)
            except Exception as e:
                print(f"⚠️  Failed to initialize Google GenAI provider: {e}")
    
    # Initialize Anthropic provider
    if AnthropicProvider is not None:
        anthropic_api_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("CLAUDE_API_KEY")
        if anthropic_api_key:
            try:
                anthropic_provider = AnthropicProvider(api_key=anthropic_api_key)
                manager.register_provider(anthropic_provider)
            except Exception as e:
                print(f"⚠️  Failed to initialize Anthropic provider: {e}")
    
    # Initialize OpenAI provider
    if OpenAIProvider is not None:
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if openai_api_key:
            try:
                openai_provider = OpenAIProvider(api_key=openai_api_key)
                manager.register_provider(openai_provider, is_default=True)  # Set as default
            except Exception as e:
                print(f"⚠️  Failed to initialize OpenAI provider: {e}")
    
    # Initialize Hugging Face provider
    if HuggingFaceProvider is not None:
        # HF provider works without API key for public models
        hf_api_key = os.getenv("HUGGINGFACE_API_KEY") or os.getenv("HF_TOKEN")
        try:
            hf_provider = HuggingFaceProvider(api_key=hf_api_key)
            manager.register_provider(hf_provider)
        except Exception as e:
            print(f"⚠️  Failed to initialize Hugging Face provider: {e}")
    
    # If no providers were registered, print warning
    if not manager.list_providers():
        print("⚠️  No AI providers were initialized. Please set API keys in environment variables:")
        print("   GOOGLE_API_KEY or GEMINI_API_KEY (for Google Gemini)")
        print("   ANTHROPIC_API_KEY or CLAUDE_API_KEY (for Anthropic Claude)")
        print("   OPENAI_API_KEY (for OpenAI GPT)")
        print("   HUGGINGFACE_API_KEY or HF_TOKEN (optional, for Hugging Face)")
    else:
        print(f"✅ Initialized {len(manager.list_providers())} AI providers: {', '.join(manager.list_providers())}")
    
    return manager


def get_provider_manager_async() -> ProviderManager:
    """Async version of get_provider_manager for FastAPI dependency injection."""
    return get_provider_manager()