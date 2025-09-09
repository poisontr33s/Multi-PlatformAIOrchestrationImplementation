"""
Base classes for AI provider adapters.

Defines the common interface that all AI providers must implement.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, AsyncIterator, Union
from dataclasses import dataclass
from enum import Enum
import asyncio
from datetime import datetime


class ModelType(Enum):
    """Types of AI models supported."""
    TEXT_GENERATION = "text-generation"
    CHAT = "chat"
    CODE_GENERATION = "code-generation"
    EMBEDDING = "embedding"
    IMAGE_GENERATION = "image-generation"


class MessageRole(Enum):
    """Chat message roles."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"


@dataclass
class AIModel:
    """Information about an AI model."""
    id: str
    name: str
    provider: str
    model_type: ModelType
    context_length: Optional[int] = None
    max_output_tokens: Optional[int] = None
    pricing_per_token: Optional[float] = None
    capabilities: List[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.capabilities is None:
            self.capabilities = []
        if self.metadata is None:
            self.metadata = {}


@dataclass 
class ChatMessage:
    """A single chat message."""
    role: MessageRole
    content: str
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class GenerationRequest:
    """Request for text/chat generation."""
    messages: List[ChatMessage]
    model_id: str
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    stop_sequences: Optional[List[str]] = None
    stream: bool = False
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class GenerationResponse:
    """Response from text/chat generation."""
    content: str
    model_id: str
    usage: Dict[str, int]  # tokens used, cost, etc.
    finish_reason: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    created_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.created_at is None:
            self.created_at = datetime.now()


class AIProvider(ABC):
    """
    Abstract base class for AI providers.
    
    All AI provider implementations must inherit from this class
    and implement the required methods.
    """
    
    def __init__(self, provider_name: str, api_key: Optional[str] = None, **kwargs):
        """Initialize the provider."""
        self.provider_name = provider_name
        self.api_key = api_key
        self.config = kwargs
        self._models_cache: Optional[List[AIModel]] = None
        self._models_cache_time: Optional[datetime] = None
        
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the provider (authenticate, setup client, etc.)."""
        pass
    
    @abstractmethod
    async def list_models(self, force_refresh: bool = False) -> List[AIModel]:
        """List all available models from this provider."""
        pass
    
    @abstractmethod
    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Generate text/chat response."""
        pass
    
    @abstractmethod
    async def generate_stream(self, request: GenerationRequest) -> AsyncIterator[str]:
        """Generate streaming text/chat response."""
        pass
    
    async def get_model(self, model_id: str) -> Optional[AIModel]:
        """Get information about a specific model."""
        models = await self.list_models()
        for model in models:
            if model.id == model_id:
                return model
        return None
    
    async def health_check(self) -> bool:
        """Check if the provider is healthy and accessible."""
        try:
            models = await self.list_models()
            return len(models) > 0
        except Exception:
            return False
    
    def _should_refresh_cache(self, max_age_minutes: int = 30) -> bool:
        """Check if model cache should be refreshed."""
        if self._models_cache is None or self._models_cache_time is None:
            return True
        
        age = datetime.now() - self._models_cache_time
        return age.total_seconds() > (max_age_minutes * 60)
    
    def _update_cache(self, models: List[AIModel]) -> None:
        """Update the models cache."""
        self._models_cache = models
        self._models_cache_time = datetime.now()
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(provider='{self.provider_name}')"


class ProviderManager:
    """
    Manager for multiple AI providers.
    
    Allows unified access to different providers and load balancing.
    """
    
    def __init__(self):
        self._providers: Dict[str, AIProvider] = {}
        self._default_provider: Optional[str] = None
    
    def register_provider(self, provider: AIProvider, is_default: bool = False) -> None:
        """Register a new provider."""
        self._providers[provider.provider_name] = provider
        if is_default or len(self._providers) == 1:
            self._default_provider = provider.provider_name
    
    def get_provider(self, provider_name: Optional[str] = None) -> Optional[AIProvider]:
        """Get a provider by name, or the default provider."""
        if provider_name is None:
            provider_name = self._default_provider
        return self._providers.get(provider_name)
    
    async def list_all_models(self) -> Dict[str, List[AIModel]]:
        """List models from all registered providers."""
        all_models = {}
        for name, provider in self._providers.items():
            try:
                models = await provider.list_models()
                all_models[name] = models
            except Exception as e:
                # Log error but continue with other providers
                all_models[name] = []
        return all_models
    
    async def generate(self, request: GenerationRequest, provider_name: Optional[str] = None) -> GenerationResponse:
        """Generate using specified provider or default."""
        provider = self.get_provider(provider_name)
        if provider is None:
            raise ValueError(f"Provider '{provider_name}' not found")
        
        return await provider.generate(request)
    
    async def health_check_all(self) -> Dict[str, bool]:
        """Check health of all providers."""
        health_status = {}
        for name, provider in self._providers.items():
            health_status[name] = await provider.health_check()
        return health_status
    
    def list_providers(self) -> List[str]:
        """List all registered provider names."""
        return list(self._providers.keys())
    
    @property
    def default_provider(self) -> Optional[str]:
        """Get the default provider name."""
        return self._default_provider
    
    @default_provider.setter  
    def default_provider(self, provider_name: str) -> None:
        """Set the default provider."""
        if provider_name not in self._providers:
            raise ValueError(f"Provider '{provider_name}' not registered")
        self._default_provider = provider_name