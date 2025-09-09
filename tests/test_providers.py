"""
Tests for AI provider adapters.
"""

import pytest
from ai_orchestration.providers import (
    AIProvider, AIModel, ChatMessage, GenerationRequest, GenerationResponse,
    ProviderManager, MessageRole, ModelType
)

# Test that base classes work without provider SDKs
def test_ai_model_creation():
    """Test AIModel creation."""
    model = AIModel(
        id="test-model",
        name="Test Model",
        provider="test",
        model_type=ModelType.CHAT,
        context_length=4096,
        capabilities=["chat", "reasoning"]
    )
    
    assert model.id == "test-model"
    assert model.name == "Test Model"
    assert model.provider == "test"
    assert model.model_type == ModelType.CHAT
    assert model.context_length == 4096
    assert "chat" in model.capabilities
    assert "reasoning" in model.capabilities


def test_chat_message_creation():
    """Test ChatMessage creation."""
    message = ChatMessage(
        role=MessageRole.USER,
        content="Hello, world!",
        metadata={"timestamp": "2024-01-01"}
    )
    
    assert message.role == MessageRole.USER
    assert message.content == "Hello, world!"
    assert message.metadata["timestamp"] == "2024-01-01"


def test_generation_request_creation():
    """Test GenerationRequest creation."""
    messages = [
        ChatMessage(role=MessageRole.USER, content="Hello"),
        ChatMessage(role=MessageRole.ASSISTANT, content="Hi there!")
    ]
    
    request = GenerationRequest(
        messages=messages,
        model_id="test-model",
        max_tokens=100,
        temperature=0.7
    )
    
    assert len(request.messages) == 2
    assert request.model_id == "test-model"
    assert request.max_tokens == 100
    assert request.temperature == 0.7


def test_generation_response_creation():
    """Test GenerationResponse creation."""
    response = GenerationResponse(
        content="Hello back!",
        model_id="test-model",
        usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    )
    
    assert response.content == "Hello back!"
    assert response.model_id == "test-model"
    assert response.usage["total_tokens"] == 15
    assert response.created_at is not None


def test_provider_manager():
    """Test ProviderManager functionality."""
    manager = ProviderManager()
    
    # Mock provider for testing
    class MockProvider(AIProvider):
        def __init__(self):
            super().__init__("mock")
            
        async def initialize(self):
            pass
            
        async def list_models(self, force_refresh=False):
            return [
                AIModel(
                    id="mock-model",
                    name="Mock Model",
                    provider="mock",
                    model_type=ModelType.CHAT,
                    capabilities=["chat"]
                )
            ]
            
        async def generate(self, request):
            return GenerationResponse(
                content="Mock response",
                model_id=request.model_id,
                usage={"total_tokens": 10}
            )
            
        async def generate_stream(self, request):
            yield "Mock"
            yield " stream"
    
    mock_provider = MockProvider()
    manager.register_provider(mock_provider, is_default=True)
    
    assert len(manager.list_providers()) == 1
    assert "mock" in manager.list_providers()
    assert manager.default_provider == "mock"
    assert manager.get_provider("mock") == mock_provider
    assert manager.get_provider() == mock_provider  # default


@pytest.mark.asyncio
async def test_provider_manager_operations():
    """Test async ProviderManager operations."""
    manager = ProviderManager()
    
    # Mock provider for testing
    class MockProvider(AIProvider):
        def __init__(self):
            super().__init__("mock")
            
        async def initialize(self):
            pass
            
        async def list_models(self, force_refresh=False):
            return [
                AIModel(
                    id="mock-model",
                    name="Mock Model", 
                    provider="mock",
                    model_type=ModelType.CHAT,
                    capabilities=["chat"]
                )
            ]
            
        async def generate(self, request):
            return GenerationResponse(
                content="Mock response",
                model_id=request.model_id,
                usage={"total_tokens": 10}
            )
            
        async def generate_stream(self, request):
            yield "Mock"
            yield " stream"
            
        async def health_check(self):
            return True
    
    mock_provider = MockProvider()
    manager.register_provider(mock_provider)
    
    # Test list_all_models
    all_models = await manager.list_all_models()
    assert "mock" in all_models
    assert len(all_models["mock"]) == 1
    assert all_models["mock"][0].id == "mock-model"
    
    # Test health_check_all
    health = await manager.health_check_all()
    assert health["mock"] is True
    
    # Test generate
    request = GenerationRequest(
        messages=[ChatMessage(role=MessageRole.USER, content="Test")],
        model_id="mock-model"
    )
    
    response = await manager.generate(request, "mock")
    assert response.content == "Mock response"
    assert response.model_id == "mock-model"


# Test that provider imports work (with optional dependencies)
def test_provider_imports():
    """Test that provider imports work correctly."""
    from ai_orchestration.providers import (
        GoogleGenAIProvider, AnthropicProvider, OpenAIProvider, HuggingFaceProvider
    )
    
    # These may be None if dependencies aren't installed, which is fine
    # We just test that the imports don't fail
    providers = [GoogleGenAIProvider, AnthropicProvider, OpenAIProvider, HuggingFaceProvider]
    
    # At least the base classes should be available
    assert AIProvider is not None
    assert ProviderManager is not None
    assert MessageRole is not None
    assert ModelType is not None