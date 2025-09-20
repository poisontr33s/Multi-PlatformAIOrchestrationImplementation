"""
Pydantic models for the AI Orchestration API.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from enum import Enum
from datetime import datetime

from ..providers import AIModel as CoreAIModel


class MessageRole(str, Enum):
    """Chat message roles."""
    system = "system"
    user = "user"
    assistant = "assistant"
    function = "function"


class ChatMessage(BaseModel):
    """A chat message."""
    role: MessageRole = Field(..., description="The role of the message author")
    content: str = Field(..., description="The content of the message")


class ModelInfo(BaseModel):
    """Information about an AI model."""
    id: str = Field(..., description="Unique model identifier")
    name: str = Field(..., description="Human-readable model name")
    provider: str = Field(..., description="Provider name (google-genai, anthropic, openai, huggingface)")
    type: str = Field(..., description="Model type (chat, text-generation, etc.)")
    context_length: Optional[int] = Field(None, description="Maximum context length in tokens")
    max_output_tokens: Optional[int] = Field(None, description="Maximum output tokens")
    capabilities: List[str] = Field(default_factory=list, description="Model capabilities")
    pricing_per_token: Optional[float] = Field(None, description="Cost per token (if available)")
    
    @classmethod
    def from_ai_model(cls, model: CoreAIModel) -> "ModelInfo":
        """Convert from core AIModel to API ModelInfo."""
        return cls(
            id=model.id,
            name=model.name,
            provider=model.provider,
            type=model.model_type.value,
            context_length=model.context_length,
            max_output_tokens=model.max_output_tokens,
            capabilities=model.capabilities,
            pricing_per_token=model.pricing_per_token
        )


class GenerateRequest(BaseModel):
    """Request for text generation."""
    model: str = Field(..., description="Model ID to use for generation")
    prompt: str = Field(..., description="The prompt to generate completion for")
    provider: Optional[str] = Field(None, description="Specific provider to use (optional)")
    system: Optional[str] = Field(None, description="System prompt")
    max_tokens: Optional[int] = Field(None, description="Maximum tokens to generate", ge=1, le=32000)
    temperature: Optional[float] = Field(None, description="Sampling temperature", ge=0.0, le=2.0)
    top_p: Optional[float] = Field(None, description="Nucleus sampling parameter", ge=0.0, le=1.0)
    stop: Optional[List[str]] = Field(None, description="Stop sequences")
    stream: bool = Field(False, description="Enable streaming response")


class GenerateResponse(BaseModel):
    """Response from text generation."""
    text: str = Field(..., description="Generated text")
    model: str = Field(..., description="Model that generated the response")
    usage: Dict[str, int] = Field(..., description="Token usage information")
    finish_reason: Optional[str] = Field(None, description="Reason generation stopped")


class ChatRequest(BaseModel):
    """Request for chat completion."""
    model: str = Field(..., description="Model ID to use for chat completion")
    messages: List[ChatMessage] = Field(..., description="Conversation history")
    provider: Optional[str] = Field(None, description="Specific provider to use (optional)")
    max_tokens: Optional[int] = Field(None, description="Maximum tokens to generate", ge=1, le=32000)
    temperature: Optional[float] = Field(None, description="Sampling temperature", ge=0.0, le=2.0)
    top_p: Optional[float] = Field(None, description="Nucleus sampling parameter", ge=0.0, le=1.0)
    stop: Optional[List[str]] = Field(None, description="Stop sequences")
    stream: bool = Field(False, description="Enable streaming response")


class ChatResponse(BaseModel):
    """Response from chat completion."""
    message: Dict[str, str] = Field(..., description="Generated message")
    model: str = Field(..., description="Model that generated the response")
    usage: Dict[str, int] = Field(..., description="Token usage information")
    finish_reason: Optional[str] = Field(None, description="Reason generation stopped")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Overall health status (healthy, degraded, unhealthy)")
    providers: Dict[str, bool] = Field(..., description="Health status of each provider")
    timestamp: datetime = Field(default_factory=datetime.now, description="Timestamp of health check")


class ErrorResponse(BaseModel):
    """Error response."""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    code: int = Field(..., description="HTTP status code")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")