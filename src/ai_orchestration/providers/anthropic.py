"""
Anthropic Claude provider implementation.
"""

from typing import List, Optional, AsyncIterator, Dict, Any
import asyncio
import os
from datetime import datetime

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    # Create mock for type hints when package not available
    class MockAnthropic:
        AsyncAnthropic = object
    anthropic = MockAnthropic()

from .base import (
    AIProvider, AIModel, ChatMessage, GenerationRequest, GenerationResponse,
    ModelType, MessageRole
)


class AnthropicProvider(AIProvider):
    """Anthropic Claude provider."""
    
    def __init__(self, api_key: Optional[str] = None, **kwargs):
        super().__init__("anthropic", api_key, **kwargs)
        self.client: Optional[anthropic.AsyncAnthropic] = None
        
        # Get API key from environment if not provided
        if self.api_key is None:
            self.api_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("CLAUDE_API_KEY")
        
        if not ANTHROPIC_AVAILABLE:
            raise ImportError(
                "anthropic package not available. "
                "Install with: pip install anthropic"
            )
    
    async def initialize(self) -> None:
        """Initialize the Anthropic client."""
        if self.api_key is None:
            raise ValueError(
                "Anthropic API key required. Set ANTHROPIC_API_KEY or CLAUDE_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        self.client = anthropic.AsyncAnthropic(api_key=self.api_key)
    
    async def list_models(self, force_refresh: bool = False) -> List[AIModel]:
        """List available Anthropic models."""
        if not force_refresh and not self._should_refresh_cache():
            return self._models_cache or []
        
        # Anthropic doesn't have a models endpoint, so we define known models
        models = [
            AIModel(
                id="claude-3-5-sonnet-20241022",
                name="Claude 3.5 Sonnet",
                provider=self.provider_name,
                model_type=ModelType.CHAT,
                context_length=200000,
                max_output_tokens=8192,
                capabilities=["text-generation", "chat", "analysis", "reasoning"],
                metadata={
                    "description": "Most intelligent model, best for complex reasoning",
                    "version": "3.5",
                    "release_date": "2024-10-22"
                }
            ),
            AIModel(
                id="claude-3-5-haiku-20241022",
                name="Claude 3.5 Haiku", 
                provider=self.provider_name,
                model_type=ModelType.CHAT,
                context_length=200000,
                max_output_tokens=8192,
                capabilities=["text-generation", "chat", "fast-response"],
                metadata={
                    "description": "Fast and efficient model for everyday tasks",
                    "version": "3.5",
                    "release_date": "2024-10-22"
                }
            ),
            AIModel(
                id="claude-3-opus-20240229",
                name="Claude 3 Opus",
                provider=self.provider_name,
                model_type=ModelType.CHAT,
                context_length=200000,
                max_output_tokens=4096,
                capabilities=["text-generation", "chat", "analysis", "creative-writing"],
                metadata={
                    "description": "Powerful model for highly complex tasks",
                    "version": "3.0",
                    "release_date": "2024-02-29"
                }
            ),
            AIModel(
                id="claude-3-sonnet-20240229",
                name="Claude 3 Sonnet",
                provider=self.provider_name,
                model_type=ModelType.CHAT,
                context_length=200000,
                max_output_tokens=4096,
                capabilities=["text-generation", "chat", "analysis"],
                metadata={
                    "description": "Balanced model for a wide range of tasks",
                    "version": "3.0",
                    "release_date": "2024-02-29"
                }
            ),
            AIModel(
                id="claude-3-haiku-20240307",
                name="Claude 3 Haiku",
                provider=self.provider_name,
                model_type=ModelType.CHAT,
                context_length=200000,
                max_output_tokens=4096,
                capabilities=["text-generation", "chat", "fast-response"],
                metadata={
                    "description": "Fast and lightweight model",
                    "version": "3.0",
                    "release_date": "2024-03-07"
                }
            ),
        ]
        
        self._update_cache(models)
        return models
    
    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Generate text using Anthropic Claude."""
        if self.client is None:
            await self.initialize()
        
        try:
            # Convert messages to Anthropic format
            anthropic_messages = self._convert_messages(request.messages)
            
            # Prepare request parameters
            params = {
                "model": request.model_id,
                "messages": anthropic_messages,
                "max_tokens": request.max_tokens or 4096,
            }
            
            if request.temperature is not None:
                params["temperature"] = request.temperature
            if request.top_p is not None:
                params["top_p"] = request.top_p
            if request.stop_sequences:
                params["stop_sequences"] = request.stop_sequences
            
            # Generate content
            response = await self.client.messages.create(**params)
            
            # Extract response content
            content = ""
            for content_block in response.content:
                if content_block.type == "text":
                    content += content_block.text
            
            # Extract usage information
            usage = {
                'prompt_tokens': response.usage.input_tokens,
                'completion_tokens': response.usage.output_tokens,
                'total_tokens': response.usage.input_tokens + response.usage.output_tokens,
            }
            
            return GenerationResponse(
                content=content,
                model_id=request.model_id,
                usage=usage,
                finish_reason=response.stop_reason,
                metadata={
                    'provider': self.provider_name,
                    'stop_sequence': response.stop_sequence
                }
            )
            
        except Exception as e:
            raise RuntimeError(f"Anthropic generation failed: {str(e)}")
    
    async def generate_stream(self, request: GenerationRequest) -> AsyncIterator[str]:
        """Generate streaming text using Anthropic Claude."""
        if self.client is None:
            await self.initialize()
        
        try:
            # Convert messages to Anthropic format
            anthropic_messages = self._convert_messages(request.messages)
            
            # Prepare request parameters
            params = {
                "model": request.model_id,
                "messages": anthropic_messages,
                "max_tokens": request.max_tokens or 4096,
                "stream": True,
            }
            
            if request.temperature is not None:
                params["temperature"] = request.temperature
            if request.top_p is not None:
                params["top_p"] = request.top_p
            if request.stop_sequences:
                params["stop_sequences"] = request.stop_sequences
            
            # Generate streaming content
            async with self.client.messages.stream(**params) as stream:
                async for text in stream.text_stream:
                    yield text
                    
        except Exception as e:
            raise RuntimeError(f"Anthropic streaming failed: {str(e)}")
    
    def _convert_messages(self, messages: List[ChatMessage]) -> List[Dict[str, Any]]:
        """Convert ChatMessage list to Anthropic format."""
        anthropic_messages = []
        system_message = None
        
        for message in messages:
            if message.role == MessageRole.SYSTEM:
                # Anthropic handles system messages separately
                system_message = message.content
            elif message.role == MessageRole.USER:
                anthropic_messages.append({
                    "role": "user",
                    "content": message.content
                })
            elif message.role == MessageRole.ASSISTANT:
                anthropic_messages.append({
                    "role": "assistant", 
                    "content": message.content
                })
            # Skip function messages as Anthropic doesn't support them directly
        
        # If we have a system message, prepend it to the first user message
        if system_message and anthropic_messages:
            first_msg = anthropic_messages[0]
            if first_msg["role"] == "user":
                first_msg["content"] = f"{system_message}\n\n{first_msg['content']}"
            else:
                # Insert system message as first user message
                anthropic_messages.insert(0, {
                    "role": "user",
                    "content": system_message
                })
        
        return anthropic_messages
    
    async def health_check(self) -> bool:
        """Check if Anthropic is accessible."""
        try:
            if self.client is None:
                await self.initialize()
            
            # Try a simple generation as a health check
            test_message = [{"role": "user", "content": "Hello"}]
            
            response = await self.client.messages.create(
                model="claude-3-haiku-20240307",  # Use fastest model for health check
                messages=test_message,
                max_tokens=10
            )
            
            return response is not None and len(response.content) > 0
            
        except Exception:
            return False