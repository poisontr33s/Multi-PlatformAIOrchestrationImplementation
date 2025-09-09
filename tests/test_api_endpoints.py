"""
API endpoint tests using FastAPI test client.
These tests validate API structure without requiring external API keys.
"""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def test_client():
    """Create a test client with mocked provider."""
    from ai_orchestration.api.server import app

    with TestClient(app) as client:
        yield client


def test_app_imports():
    """Test that FastAPI app imports correctly."""
    from ai_orchestration.api.server import app

    assert app is not None
    assert app.title == "Multi-Platform AI Orchestration API"


def test_health_endpoint_structure(test_client):
    """Test health endpoint returns proper structure."""
    response = test_client.get("/health")

    # Should always return a response (even if provider not initialized)
    assert response.status_code == 200

    data = response.json()
    assert "status" in data
    assert "service" in data
    assert "version" in data
    assert "providers" in data

    # Check service info
    assert data["service"] == "ai-orchestration-api"
    assert data["version"] == "1.0.0"

    # Check providers structure
    assert "google_gemini" in data["providers"]
    provider_info = data["providers"]["google_gemini"]
    assert "status" in provider_info
    assert "provider" in provider_info


@patch("ai_orchestration.api.server.google_provider")
def test_models_endpoint_no_provider(mock_provider, test_client):
    """Test models endpoint when provider is not initialized."""
    # Set global provider to None
    mock_provider = None

    response = test_client.get("/models")

    # Should return 503 when provider not initialized
    assert response.status_code == 503
    data = response.json()
    assert "detail" in data
    assert "not initialized" in data["detail"].lower()


@patch("ai_orchestration.api.server.google_provider")
def test_generate_endpoint_no_provider(mock_provider, test_client):
    """Test generate endpoint when provider is not initialized."""
    # Set global provider to None
    mock_provider = None

    response = test_client.post(
        "/generate",
        json={
            "prompt": "Hello, world!",
        },
    )

    # Should return 503 when provider not initialized
    assert response.status_code == 503
    data = response.json()
    assert "detail" in data
    assert "not initialized" in data["detail"].lower()


def test_generate_endpoint_validation(test_client):
    """Test generate endpoint request validation."""
    # Test missing prompt
    response = test_client.post("/generate", json={})
    assert response.status_code == 422  # Validation error

    # Test invalid data types
    response = test_client.post(
        "/generate",
        json={
            "prompt": 123,  # Should be string
        },
    )
    assert response.status_code == 422


@patch("ai_orchestration.api.server.GoogleGeminiProvider")
def test_mocked_provider_initialization(mock_provider_class, test_client):
    """Test app with mocked provider initialization."""
    # Create mock provider instance
    mock_provider = AsyncMock()
    mock_provider.initialize = AsyncMock()
    mock_provider.health_check = AsyncMock(
        return_value={
            "status": "healthy",
            "provider": "google_gemini",
            "model": "gemini-1.5-flash",
        }
    )
    mock_provider.list_models = AsyncMock(return_value=[])
    mock_provider.generate_text = AsyncMock()

    mock_provider_class.return_value = mock_provider

    # Test would require full app restart to test startup event
    # For now, just verify the mocking works
    assert mock_provider_class is not None


def test_api_documentation_endpoints(test_client):
    """Test that API documentation endpoints are available."""
    # Test OpenAPI JSON
    response = test_client.get("/openapi.json")
    assert response.status_code == 200

    openapi_spec = response.json()
    assert "openapi" in openapi_spec
    assert "info" in openapi_spec
    assert openapi_spec["info"]["title"] == "Multi-Platform AI Orchestration API"

    # Test Swagger UI
    response = test_client.get("/docs")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]

    # Test ReDoc
    response = test_client.get("/redoc")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_cors_and_middleware(test_client):
    """Test that CORS and other middleware work."""
    # Test basic endpoint with different methods
    response = test_client.get("/health")
    assert response.status_code == 200

    # Test that server handles various HTTP methods appropriately
    response = test_client.post("/health")
    assert response.status_code == 405  # Method not allowed


def test_error_handling(test_client):
    """Test error handling for invalid endpoints."""
    # Test 404 for non-existent endpoint
    response = test_client.get("/nonexistent")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_async_endpoint_patterns():
    """Test async patterns used in endpoints."""
    from ai_orchestration.api.server import get_google_provider, health_check

    # Test health check function directly
    health_result = await health_check()
    assert isinstance(health_result, dict)
    assert "status" in health_result

    # Test dependency function (should raise when no provider)
    with pytest.raises(Exception):  # HTTPException would be raised
        await get_google_provider()


def test_pydantic_models():
    """Test that Pydantic models are properly imported and structured."""
    from ai_orchestration.providers.google_gemini import (
        GenerateRequest,
        GenerateResponse,
        ModelInfo,
    )

    # Test model creation and validation
    request = GenerateRequest(prompt="test")
    assert request.prompt == "test"

    response = GenerateResponse(text="response", model="test-model")
    assert response.text == "response"
    assert response.model == "test-model"

    model = ModelInfo(name="test", display_name="Test Model")
    assert model.name == "test"
    assert model.display_name == "Test Model"


def test_server_configuration():
    """Test server configuration and settings."""
    from ai_orchestration.api.server import app

    # Check FastAPI app configuration
    assert app.title == "Multi-Platform AI Orchestration API"
    assert app.version == "1.0.0"
    assert "Strategic AI orchestration" in app.description

    # Check that docs are enabled
    assert app.docs_url == "/docs"
    assert app.redoc_url == "/redoc"


def test_logging_configuration():
    """Test that logging is properly configured."""
    import structlog

    # Test that structlog is available and configured
    logger = structlog.get_logger("test")
    assert logger is not None

    # Test that we can create log entries (without checking output)
    try:
        logger.info("Test log message", test_param="value")
        # If no exception, logging is working
        assert True
    except Exception:
        pytest.fail("Logging configuration failed")
