"""
Tests for the AI Orchestration API.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch

from ai_orchestration.api import app
from ai_orchestration.api.providers import get_provider_manager
from ai_orchestration.providers import ProviderManager, AIModel, ModelType


@pytest.fixture
def mock_provider_manager():
    """Mock provider manager for testing."""
    manager = Mock(spec=ProviderManager)
    
    # Mock models
    mock_models = [
        AIModel(
            id="gpt-4",
            name="GPT-4",
            provider="openai",
            model_type=ModelType.CHAT,
            context_length=8192,
            max_output_tokens=4096,
            capabilities=["chat", "reasoning"]
        ),
        AIModel(
            id="claude-3-5-sonnet-20241022",
            name="Claude 3.5 Sonnet",
            provider="anthropic",
            model_type=ModelType.CHAT,
            context_length=200000,
            max_output_tokens=8192,
            capabilities=["chat", "analysis"]
        )
    ]
    
    manager.list_providers.return_value = ["openai", "anthropic"]
    manager.default_provider = "openai"
    manager.list_all_models.return_value = {
        "openai": [mock_models[0]],
        "anthropic": [mock_models[1]]
    }
    manager.health_check_all.return_value = {
        "openai": True,
        "anthropic": True
    }
    
    return manager


@pytest.fixture
def client(mock_provider_manager):
    """Test client with mocked dependencies."""
    # Override the dependency
    app.dependency_overrides[get_provider_manager] = lambda: mock_provider_manager
    
    with TestClient(app) as client:
        yield client
    
    # Clean up
    app.dependency_overrides.clear()


def test_root_endpoint(client):
    """Test the root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "Multi-Platform AI Orchestration API"
    assert data["version"] == "1.0.0"


def test_health_endpoint(client):
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "providers" in data
    assert "openai" in data["providers"]
    assert "anthropic" in data["providers"]
    assert data["providers"]["openai"] is True
    assert data["providers"]["anthropic"] is True


def test_list_models_endpoint(client):
    """Test the list models endpoint."""
    response = client.get("/models")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2
    
    # Check first model
    model1 = data[0]
    assert model1["id"] == "gpt-4"
    assert model1["name"] == "GPT-4"
    assert model1["provider"] == "openai"
    assert model1["type"] == "chat"
    
    # Check second model
    model2 = data[1]
    assert model2["id"] == "claude-3-5-sonnet-20241022"
    assert model2["name"] == "Claude 3.5 Sonnet"
    assert model2["provider"] == "anthropic"


def test_list_providers_endpoint(client):
    """Test the list providers endpoint."""
    response = client.get("/providers")
    assert response.status_code == 200
    data = response.json()
    assert data["providers"] == ["openai", "anthropic"]
    assert data["default"] == "openai"
    assert data["count"] == 2


def test_generate_endpoint_validation(client):
    """Test the generate endpoint with invalid input."""
    # Missing required fields
    response = client.post("/generate", json={})
    assert response.status_code == 422  # Validation error
    
    # Invalid temperature
    response = client.post("/generate", json={
        "model": "gpt-4",
        "prompt": "Hello",
        "temperature": 3.0  # Too high
    })
    assert response.status_code == 422


def test_chat_endpoint_validation(client):
    """Test the chat endpoint with invalid input."""
    # Missing required fields
    response = client.post("/chat", json={})
    assert response.status_code == 422
    
    # Invalid message format
    response = client.post("/chat", json={
        "model": "gpt-4",
        "messages": [
            {"role": "invalid_role", "content": "Hello"}
        ]
    })
    assert response.status_code == 422


def test_openapi_docs(client):
    """Test that OpenAPI documentation is available."""
    response = client.get("/openapi.json")
    assert response.status_code == 200
    
    openapi_spec = response.json()
    assert openapi_spec["info"]["title"] == "Multi-Platform AI Orchestration API"
    assert openapi_spec["info"]["version"] == "1.0.0"
    
    # Check that our endpoints are documented
    assert "/models" in openapi_spec["paths"]
    assert "/generate" in openapi_spec["paths"]
    assert "/chat" in openapi_spec["paths"]
    assert "/health" in openapi_spec["paths"]


def test_swagger_ui_redirect(client):
    """Test that Swagger UI is accessible."""
    response = client.get("/docs")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]