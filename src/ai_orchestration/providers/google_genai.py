"""
Google GenAI provider implementation using the new google-genai SDK.

This replaces the deprecated Google Generative AI client.
"""

from typing import List, Optional, AsyncIterator, Dict, Any
import asyncio
import os
from datetime import datetime

try:
    import google.genai
    from google.genai import types
    GOOGLE_GENAI_AVAILABLE = True
except ImportError:
    GOOGLE_GENAI_AVAILABLE = False
    # Create mock types for type hints when package not available
    class MockTypes:
        Content = object
        Part = object
        GenerateContentConfig = object
    types = MockTypes()

from .base import (
    AIProvider, AIModel, ChatMessage, GenerationRequest, GenerationResponse,
    ModelType, MessageRole
)


class GoogleGenAIProvider(AIProvider):
    """Google GenAI provider using the new google-genai SDK."""
    
    def __init__(self, api_key: Optional[str] = None, **kwargs):
        super().__init__("google-genai", api_key, **kwargs)
        self.client: Optional[google.genai.Client] = None
        
        # Get API key from environment if not provided
        if self.api_key is None:
            self.api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        
        if not GOOGLE_GENAI_AVAILABLE:
            raise ImportError(
                "google-genai package not available. "
                "Install with: pip install google-genai"
            )
    
    async def initialize(self) -> None:
        """Initialize the Google GenAI client."""
        if self.api_key is None:
            raise ValueError(
                "Google API key required. Set GOOGLE_API_KEY or GEMINI_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        # Configure the client
        google.genai.configure(api_key=self.api_key)
        self.client = google.genai.Client(api_key=self.api_key)
    
    async def list_models(self, force_refresh: bool = False) -> List[AIModel]:
        """List available Google GenAI models."""
        if not force_refresh and not self._should_refresh_cache():
            return self._models_cache or []
        
        if self.client is None:
            await self.initialize()
        
        try:
            # Get available models
            models = []
            async for model in self.client.models.list():
                # Parse model capabilities and context length
                capabilities = []
                context_length = None
                max_output_tokens = None
                
                # Extract model information
                if hasattr(model, 'supported_generation_methods'):
                    if 'generateContent' in model.supported_generation_methods:
                        capabilities.append('text-generation')
                    if 'generateMessage' in model.supported_generation_methods:
                        capabilities.append('chat')
                
                if hasattr(model, 'input_token_limit'):
                    context_length = model.input_token_limit
                
                if hasattr(model, 'output_token_limit'):
                    max_output_tokens = model.output_token_limit
                
                # Determine model type
                model_type = ModelType.TEXT_GENERATION
                if 'chat' in capabilities:
                    model_type = ModelType.CHAT
                
                ai_model = AIModel(
                    id=model.name,
                    name=model.display_name if hasattr(model, 'display_name') else model.name,
                    provider=self.provider_name,
                    model_type=model_type,
                    context_length=context_length,
                    max_output_tokens=max_output_tokens,
                    capabilities=capabilities,
                    metadata={
                        'description': getattr(model, 'description', ''),
                        'version': getattr(model, 'version', ''),
                    }
                )
                models.append(ai_model)
            
            self._update_cache(models)
            return models
            
        except Exception as e:
            raise RuntimeError(f"Failed to list Google GenAI models: {str(e)}")
    
    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Generate text using Google GenAI."""
        if self.client is None:
            await self.initialize()
        
        try:
            # Convert messages to Google GenAI format
            contents = self._convert_messages_to_contents(request.messages)
            
            # Prepare generation config
            generation_config = {}
            if request.max_tokens:
                generation_config['max_output_tokens'] = request.max_tokens
            if request.temperature is not None:
                generation_config['temperature'] = request.temperature
            if request.top_p is not None:
                generation_config['top_p'] = request.top_p
            if request.stop_sequences:
                generation_config['stop_sequences'] = request.stop_sequences
            
            # Generate content
            response = await self.client.models.generate_content(
                model=request.model_id,
                contents=contents,
                config=types.GenerateContentConfig(**generation_config) if generation_config else None
            )
            
            # Extract response content
            content = ""
            if response.candidates and len(response.candidates) > 0:
                candidate = response.candidates[0]
                if candidate.content and candidate.content.parts:
                    content = "".join(part.text for part in candidate.content.parts if hasattr(part, 'text'))
            
            # Extract usage information
            usage = {}
            if hasattr(response, 'usage_metadata'):
                usage = {
                    'prompt_tokens': getattr(response.usage_metadata, 'prompt_token_count', 0),
                    'completion_tokens': getattr(response.usage_metadata, 'candidates_token_count', 0),
                    'total_tokens': getattr(response.usage_metadata, 'total_token_count', 0),
                }
            
            # Extract finish reason
            finish_reason = None
            if response.candidates and len(response.candidates) > 0:
                finish_reason = getattr(response.candidates[0], 'finish_reason', None)
            
            return GenerationResponse(
                content=content,
                model_id=request.model_id,
                usage=usage,
                finish_reason=finish_reason,
                metadata={'provider': self.provider_name}
            )
            
        except Exception as e:
            raise RuntimeError(f"Google GenAI generation failed: {str(e)}")
    
    async def generate_stream(self, request: GenerationRequest) -> AsyncIterator[str]:
        """Generate streaming text using Google GenAI."""
        if self.client is None:
            await self.initialize()
        
        try:
            # Convert messages to Google GenAI format
            contents = self._convert_messages_to_contents(request.messages)
            
            # Prepare generation config
            generation_config = {}
            if request.max_tokens:
                generation_config['max_output_tokens'] = request.max_tokens
            if request.temperature is not None:
                generation_config['temperature'] = request.temperature
            if request.top_p is not None:
                generation_config['top_p'] = request.top_p
            if request.stop_sequences:
                generation_config['stop_sequences'] = request.stop_sequences
            
            # Generate streaming content
            async for chunk in self.client.models.generate_content_stream(
                model=request.model_id,
                contents=contents,
                config=types.GenerateContentConfig(**generation_config) if generation_config else None
            ):
                if chunk.candidates and len(chunk.candidates) > 0:
                    candidate = chunk.candidates[0]
                    if candidate.content and candidate.content.parts:
                        for part in candidate.content.parts:
                            if hasattr(part, 'text') and part.text:
                                yield part.text
                                
        except Exception as e:
            raise RuntimeError(f"Google GenAI streaming failed: {str(e)}")
    
    def _convert_messages_to_contents(self, messages: List[ChatMessage]) -> List[types.Content]:
        """Convert ChatMessage list to Google GenAI Contents format."""
        contents = []
        
        for message in messages:
            # Map roles
            role = "user"  # default
            if message.role == MessageRole.ASSISTANT:
                role = "model"
            elif message.role == MessageRole.SYSTEM:
                # Google GenAI doesn't have explicit system role, prepend to user message
                role = "user"
            elif message.role == MessageRole.USER:
                role = "user"
            
            content = types.Content(
                role=role,
                parts=[types.Part(text=message.content)]
            )
            contents.append(content)
        
        return contents
    
    async def health_check(self) -> bool:
        """Check if Google GenAI is accessible."""
        try:
            if self.client is None:
                await self.initialize()
            
            # Try to list models as a health check
            models = await self.list_models()
            return len(models) > 0
            
        except Exception:
            return False