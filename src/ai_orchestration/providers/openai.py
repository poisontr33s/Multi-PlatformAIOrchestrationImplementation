"""
OpenAI provider implementation.
"""

from typing import List, Optional, AsyncIterator, Dict, Any
import asyncio
import os
from datetime import datetime

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    # Create mock for type hints when package not available
    class MockOpenAI:
        AsyncOpenAI = object
    openai = MockOpenAI()

from .base import (
    AIProvider, AIModel, ChatMessage, GenerationRequest, GenerationResponse,
    ModelType, MessageRole
)


class OpenAIProvider(AIProvider):
    """OpenAI GPT provider."""
    
    def __init__(self, api_key: Optional[str] = None, **kwargs):
        super().__init__("openai", api_key, **kwargs)
        self.client: Optional[openai.AsyncOpenAI] = None
        
        # Get API key from environment if not provided
        if self.api_key is None:
            self.api_key = os.getenv("OPENAI_API_KEY")
        
        if not OPENAI_AVAILABLE:
            raise ImportError(
                "openai package not available. "
                "Install with: pip install openai"
            )
    
    async def initialize(self) -> None:
        """Initialize the OpenAI client."""
        if self.api_key is None:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        self.client = openai.AsyncOpenAI(api_key=self.api_key)
    
    async def list_models(self, force_refresh: bool = False) -> List[AIModel]:
        """List available OpenAI models."""
        if not force_refresh and not self._should_refresh_cache():
            return self._models_cache or []
        
        if self.client is None:
            await self.initialize()
        
        try:
            # Get models from OpenAI API
            response = await self.client.models.list()
            models = []
            
            for model_data in response.data:
                # Filter for relevant models and extract info
                model_id = model_data.id
                
                # Determine model type and capabilities
                capabilities = []
                model_type = ModelType.TEXT_GENERATION
                context_length = None
                max_output_tokens = None
                
                if "gpt" in model_id:
                    capabilities.extend(["text-generation", "chat"])
                    model_type = ModelType.CHAT
                    
                    # Set context lengths for known models
                    if "gpt-4" in model_id:
                        if "turbo" in model_id or "preview" in model_id:
                            context_length = 128000
                            max_output_tokens = 4096
                        elif "32k" in model_id:
                            context_length = 32768
                            max_output_tokens = 4096
                        else:
                            context_length = 8192
                            max_output_tokens = 4096
                    elif "gpt-3.5" in model_id:
                        if "16k" in model_id:
                            context_length = 16384
                            max_output_tokens = 4096
                        else:
                            context_length = 4096
                            max_output_tokens = 4096
                    elif "o1" in model_id:
                        context_length = 200000
                        max_output_tokens = 100000
                        capabilities.append("reasoning")
                        
                elif "davinci" in model_id or "curie" in model_id or "babbage" in model_id or "ada" in model_id:
                    capabilities.append("text-generation")
                    context_length = 4096
                    max_output_tokens = 4096
                    
                elif "text-embedding" in model_id:
                    model_type = ModelType.EMBEDDING
                    capabilities.append("embedding")
                    context_length = 8191
                    
                elif "dall-e" in model_id:
                    model_type = ModelType.IMAGE_GENERATION
                    capabilities.append("image-generation")
                    
                elif "whisper" in model_id:
                    capabilities.extend(["audio-transcription", "audio-translation"])
                    
                elif "tts" in model_id:
                    capabilities.append("text-to-speech")
                    
                # Skip models we don't support
                if not capabilities:
                    continue
                
                ai_model = AIModel(
                    id=model_id,
                    name=model_id,  # OpenAI doesn't provide display names
                    provider=self.provider_name,
                    model_type=model_type,
                    context_length=context_length,
                    max_output_tokens=max_output_tokens,
                    capabilities=capabilities,
                    metadata={
                        'created': model_data.created,
                        'owned_by': model_data.owned_by,
                    }
                )
                models.append(ai_model)
            
            self._update_cache(models)
            return models
            
        except Exception as e:
            raise RuntimeError(f"Failed to list OpenAI models: {str(e)}")
    
    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Generate text using OpenAI."""
        if self.client is None:
            await self.initialize()
        
        try:
            # Convert messages to OpenAI format
            openai_messages = self._convert_messages(request.messages)
            
            # Prepare request parameters
            params = {
                "model": request.model_id,
                "messages": openai_messages,
            }
            
            if request.max_tokens:
                params["max_tokens"] = request.max_tokens
            if request.temperature is not None:
                params["temperature"] = request.temperature
            if request.top_p is not None:
                params["top_p"] = request.top_p
            if request.stop_sequences:
                params["stop"] = request.stop_sequences
            
            # Generate content
            response = await self.client.chat.completions.create(**params)
            
            # Extract response content
            content = ""
            if response.choices and len(response.choices) > 0:
                choice = response.choices[0]
                if choice.message and choice.message.content:
                    content = choice.message.content
            
            # Extract usage information
            usage = {}
            if response.usage:
                usage = {
                    'prompt_tokens': response.usage.prompt_tokens,
                    'completion_tokens': response.usage.completion_tokens,
                    'total_tokens': response.usage.total_tokens,
                }
            
            # Extract finish reason
            finish_reason = None
            if response.choices and len(response.choices) > 0:
                finish_reason = response.choices[0].finish_reason
            
            return GenerationResponse(
                content=content,
                model_id=request.model_id,
                usage=usage,
                finish_reason=finish_reason,
                metadata={
                    'provider': self.provider_name,
                    'model': response.model,
                    'created': response.created,
                }
            )
            
        except Exception as e:
            raise RuntimeError(f"OpenAI generation failed: {str(e)}")
    
    async def generate_stream(self, request: GenerationRequest) -> AsyncIterator[str]:
        """Generate streaming text using OpenAI."""
        if self.client is None:
            await self.initialize()
        
        try:
            # Convert messages to OpenAI format
            openai_messages = self._convert_messages(request.messages)
            
            # Prepare request parameters
            params = {
                "model": request.model_id,
                "messages": openai_messages,
                "stream": True,
            }
            
            if request.max_tokens:
                params["max_tokens"] = request.max_tokens
            if request.temperature is not None:
                params["temperature"] = request.temperature
            if request.top_p is not None:
                params["top_p"] = request.top_p
            if request.stop_sequences:
                params["stop"] = request.stop_sequences
            
            # Generate streaming content
            stream = await self.client.chat.completions.create(**params)
            
            async for chunk in stream:
                if chunk.choices and len(chunk.choices) > 0:
                    choice = chunk.choices[0]
                    if choice.delta and choice.delta.content:
                        yield choice.delta.content
                        
        except Exception as e:
            raise RuntimeError(f"OpenAI streaming failed: {str(e)}")
    
    def _convert_messages(self, messages: List[ChatMessage]) -> List[Dict[str, Any]]:
        """Convert ChatMessage list to OpenAI format."""
        openai_messages = []
        
        for message in messages:
            role = "user"  # default
            if message.role == MessageRole.SYSTEM:
                role = "system"
            elif message.role == MessageRole.USER:
                role = "user"
            elif message.role == MessageRole.ASSISTANT:
                role = "assistant"
            elif message.role == MessageRole.FUNCTION:
                role = "function"
            
            openai_messages.append({
                "role": role,
                "content": message.content
            })
        
        return openai_messages
    
    async def health_check(self) -> bool:
        """Check if OpenAI is accessible."""
        try:
            if self.client is None:
                await self.initialize()
            
            # Try to list models as a health check
            models = await self.list_models()
            return len(models) > 0
            
        except Exception:
            return False