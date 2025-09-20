"""
Import-only tests for Google Gemini provider.
These tests validate imports and basic structure without requiring API keys.
"""

from unittest.mock import patch

import pytest


def test_google_gemini_imports():
    """Test that Google Gemini provider imports work correctly."""
    # Test basic imports
    from ai_orchestration.providers.google_gemini import (
        GenerateRequest,
        GenerateResponse,
        GoogleGeminiProvider,
        ModelInfo,
    )

    # Verify classes exist
    assert GoogleGeminiProvider is not None
    assert GenerateRequest is not None
    assert GenerateResponse is not None
    assert ModelInfo is not None


def test_generate_request_model():
    """Test GenerateRequest model validation."""
    from ai_orchestration.providers.google_gemini import GenerateRequest

    # Test valid request
    request = GenerateRequest(prompt="Hello, world!")
    assert request.prompt == "Hello, world!"
    assert request.max_tokens is None
    assert request.temperature is None

    # Test with all parameters
    request_full = GenerateRequest(
        prompt="Test prompt",
        max_tokens=100,
        temperature=0.7,
    )
    assert request_full.prompt == "Test prompt"
    assert request_full.max_tokens == 100
    assert request_full.temperature == 0.7


def test_generate_response_model():
    """Test GenerateResponse model structure."""
    from ai_orchestration.providers.google_gemini import GenerateResponse

    response = GenerateResponse(
        text="Generated text",
        model="gemini-2.5-pro-preview-05-06",  # Updated to September 2025 default
    )
    assert response.text == "Generated text"
    assert response.model == "gemini-2.5-pro-preview-05-06"
    assert response.usage == {}

    # Test with usage info
    response_with_usage = GenerateResponse(
        text="Generated text",
        model="gemini-2.5-pro-preview-05-06",  # Updated to September 2025 default
        usage={"total_tokens": 25},
    )
    assert response_with_usage.usage["total_tokens"] == 25


def test_model_info_structure():
    """Test ModelInfo model structure."""
    from ai_orchestration.providers.google_gemini import ModelInfo

    model = ModelInfo(
        name="models/gemini-2.5-pro-preview-05-06",  # Updated to September 2025 model
        display_name="Gemini 2.5 Pro Preview",
    )
    assert model.name == "models/gemini-2.5-pro-preview-05-06"
    assert model.display_name == "Gemini 2.5 Pro Preview"
    assert model.description == ""
    assert model.input_token_limit is None
    assert model.output_token_limit is None


@patch.dict("os.environ", {"GOOGLE_API_KEY": "test_key"})
@patch("google.generativeai.configure")
def test_provider_initialization_without_api_call(mock_configure):
    """Test provider initialization without making actual API calls."""
    from ai_orchestration.providers.google_gemini import GoogleGeminiProvider

    # Mock the genai.configure call
    mock_configure.return_value = None

    # Test initialization with mocked API key
    provider = GoogleGeminiProvider()
    assert provider.api_key == "test_key"
    assert provider.model_name == "gemini-2.5-pro-preview-05-06"  # Updated September 2025 default

    # Verify configure was called
    mock_configure.assert_called_once_with(api_key="test_key")


def test_provider_initialization_missing_api_key():
    """Test provider initialization fails without API key."""
    from ai_orchestration.providers.google_gemini import GoogleGeminiProvider

    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(ValueError, match="Google API key is required"):
            GoogleGeminiProvider()


@patch.dict(
    "os.environ",
    {
        "GOOGLE_API_KEY": "test_key",
        "GOOGLE_GENAI_MODEL": "gemini-1.0-pro",
    },
)
@patch("google.generativeai.configure")
def test_provider_custom_model(mock_configure):
    """Test provider initialization with custom model."""
    from ai_orchestration.providers.google_gemini import GoogleGeminiProvider

    mock_configure.return_value = None

    provider = GoogleGeminiProvider()
    assert provider.model_name == "gemini-1.0-pro"


def test_provider_dependencies_available():
    """Test that required dependencies are available."""
    # Test google.generativeai import
    import google.generativeai as genai

    assert genai is not None

    # Test pydantic import
    from pydantic import BaseModel

    assert BaseModel is not None

    # Test structlog import
    import structlog

    assert structlog is not None


def test_provider_api_structure():
    """Test that provider has expected methods."""
    from ai_orchestration.providers.google_gemini import GoogleGeminiProvider

    # Check that class has expected methods (without calling them)
    assert hasattr(GoogleGeminiProvider, "initialize")
    assert hasattr(GoogleGeminiProvider, "generate_text")
    assert hasattr(GoogleGeminiProvider, "list_models")
    assert hasattr(GoogleGeminiProvider, "health_check")

    # Check method signatures exist
    import inspect

    init_sig = inspect.signature(GoogleGeminiProvider.initialize)
    assert "self" in init_sig.parameters

    generate_sig = inspect.signature(GoogleGeminiProvider.generate_text)
    assert "self" in generate_sig.parameters
    assert "request" in generate_sig.parameters


def test_provider_2025_model_registry():
    """Test that provider has 2025 model registry with proper models."""
    from ai_orchestration.providers.google_gemini import GoogleGeminiProvider
    
    # Check that September 2025 models are in the registry
    assert "gemini-2.5-pro-preview-05-06" in GoogleGeminiProvider.SUPPORTED_MODELS
    assert "gemini-2.5-flash-preview-05-20" in GoogleGeminiProvider.SUPPORTED_MODELS
    assert "gemini-2.0-flash-thinking-exp" in GoogleGeminiProvider.SUPPORTED_MODELS
    
    # Check model capabilities for latest 2025 model
    pro_2_5 = GoogleGeminiProvider.SUPPORTED_MODELS["gemini-2.5-pro-preview-05-06"]
    assert pro_2_5["context_window"] >= 1000000  # 1M+ tokens
    assert pro_2_5["recommended"] is True
    assert "multimodal" in pro_2_5["features"]
    assert "2025-latest" in pro_2_5["features"]


@patch.dict("os.environ", {"GOOGLE_API_KEY": "test_key"})
@patch("google.generativeai.configure")  
def test_provider_model_helper_methods(mock_configure):
    """Test helper methods for model information."""
    from ai_orchestration.providers.google_gemini import GoogleGeminiProvider
    
    mock_configure.return_value = None
    
    # Test with September 2025 recommended model
    provider = GoogleGeminiProvider(model_name="gemini-2.5-pro-preview-05-06")
    assert provider.get_model_context_window() >= 1000000
    assert provider.is_model_recommended() is True
    assert "multimodal" in provider.get_model_features()
    assert "2025-latest" in provider.get_model_features()
    
    # Test with legacy model
    provider_legacy = GoogleGeminiProvider(model_name="gemini-1.5-flash")  
    assert provider_legacy.is_model_recommended() is False
    assert "legacy" in provider_legacy.get_model_features()
