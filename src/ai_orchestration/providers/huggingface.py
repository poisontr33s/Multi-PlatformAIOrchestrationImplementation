"""
Hugging Face provider implementation for OSS models.
"""

from typing import List, Optional, AsyncIterator, Dict, Any
import asyncio
import os
from datetime import datetime

try:
    from huggingface_hub import AsyncInferenceClient
    import huggingface_hub
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False
    # Create mock for type hints when package not available
    class MockHF:
        AsyncInferenceClient = object
    AsyncInferenceClient = MockHF.AsyncInferenceClient

from .base import (
    AIProvider, AIModel, ChatMessage, GenerationRequest, GenerationResponse,
    ModelType, MessageRole
)


class HuggingFaceProvider(AIProvider):
    """Hugging Face provider for OSS models."""
    
    def __init__(self, api_key: Optional[str] = None, **kwargs):
        super().__init__("huggingface", api_key, **kwargs)
        self.client: Optional[AsyncInferenceClient] = None
        
        # Get API key from environment if not provided
        if self.api_key is None:
            self.api_key = os.getenv("HUGGINGFACE_API_KEY") or os.getenv("HF_TOKEN")
        
        if not HUGGINGFACE_AVAILABLE:
            raise ImportError(
                "huggingface_hub package not available. "
                "Install with: pip install huggingface_hub"
            )
    
    async def initialize(self) -> None:
        """Initialize the Hugging Face client."""
        # API key is optional for Hugging Face (can use public models)
        self.client = AsyncInferenceClient(token=self.api_key)
    
    async def list_models(self, force_refresh: bool = False) -> List[AIModel]:
        """List popular/recommended Hugging Face models."""
        if not force_refresh and not self._should_refresh_cache():
            return self._models_cache or []
        
        # Define curated list of popular OSS models
        # In production, this could be enhanced to query HF Hub API
        models = [
            # Meta Llama models
            AIModel(
                id="meta-llama/Llama-3.2-3B-Instruct",
                name="Llama 3.2 3B Instruct",
                provider=self.provider_name,
                model_type=ModelType.CHAT,
                context_length=131072,
                max_output_tokens=8192,
                capabilities=["text-generation", "chat", "instruction-following"],
                metadata={
                    "description": "Latest Llama model optimized for instruction following",
                    "organization": "Meta",
                    "license": "llama3.2",
                    "size": "3B parameters"
                }
            ),
            AIModel(
                id="meta-llama/Llama-3.2-1B-Instruct",
                name="Llama 3.2 1B Instruct",
                provider=self.provider_name,
                model_type=ModelType.CHAT,
                context_length=131072,
                max_output_tokens=8192,
                capabilities=["text-generation", "chat", "fast-inference"],
                metadata={
                    "description": "Compact Llama model for fast inference",
                    "organization": "Meta",
                    "license": "llama3.2",
                    "size": "1B parameters"
                }
            ),
            
            # Google Gemma models
            AIModel(
                id="google/gemma-2-9b-it",
                name="Gemma 2 9B Instruct",
                provider=self.provider_name,
                model_type=ModelType.CHAT,
                context_length=8192,
                max_output_tokens=4096,
                capabilities=["text-generation", "chat", "reasoning"],
                metadata={
                    "description": "Google's Gemma 2 instruction-tuned model",
                    "organization": "Google",
                    "license": "gemma",
                    "size": "9B parameters"
                }
            ),
            AIModel(
                id="google/gemma-2-2b-it", 
                name="Gemma 2 2B Instruct",
                provider=self.provider_name,
                model_type=ModelType.CHAT,
                context_length=8192,
                max_output_tokens=4096,
                capabilities=["text-generation", "chat", "lightweight"],
                metadata={
                    "description": "Lightweight Gemma 2 model",
                    "organization": "Google",
                    "license": "gemma",
                    "size": "2B parameters"
                }
            ),
            
            # Microsoft Phi models
            AIModel(
                id="microsoft/Phi-3.5-mini-instruct",
                name="Phi-3.5 Mini Instruct",
                provider=self.provider_name,
                model_type=ModelType.CHAT,
                context_length=131072,
                max_output_tokens=4096,
                capabilities=["text-generation", "chat", "reasoning", "efficient"],
                metadata={
                    "description": "Microsoft's efficient small language model",
                    "organization": "Microsoft",
                    "license": "mit",
                    "size": "3.8B parameters"
                }
            ),
            
            # Mistral models
            AIModel(
                id="mistralai/Mistral-7B-Instruct-v0.3",
                name="Mistral 7B Instruct v0.3",
                provider=self.provider_name,
                model_type=ModelType.CHAT,
                context_length=32768,
                max_output_tokens=8192,
                capabilities=["text-generation", "chat", "multilingual"],
                metadata={
                    "description": "Mistral's instruction-tuned model",
                    "organization": "Mistral AI",
                    "license": "apache-2.0",
                    "size": "7B parameters"
                }
            ),
            
            # Code generation models
            AIModel(
                id="bigcode/starcoder2-15b",
                name="StarCoder2 15B",
                provider=self.provider_name,
                model_type=ModelType.CODE_GENERATION,
                context_length=16384,
                max_output_tokens=8192,
                capabilities=["code-generation", "programming", "multiple-languages"],
                metadata={
                    "description": "Advanced code generation model",
                    "organization": "BigCode",
                    "license": "bigcode-openrail-m",
                    "size": "15B parameters"
                }
            ),
            
            # Embedding models
            AIModel(
                id="sentence-transformers/all-MiniLM-L6-v2",
                name="All-MiniLM-L6-v2",
                provider=self.provider_name,
                model_type=ModelType.EMBEDDING,
                context_length=512,
                capabilities=["embedding", "semantic-search", "similarity"],
                metadata={
                    "description": "Lightweight sentence embedding model",
                    "organization": "Sentence Transformers",
                    "license": "apache-2.0",
                    "dimensions": 384
                }
            ),
        ]
        
        self._update_cache(models)
        return models
    
    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Generate text using Hugging Face."""
        if self.client is None:
            await self.initialize()
        
        try:
            # Convert messages to prompt format for HF models
            prompt = self._convert_messages_to_prompt(request.messages)
            
            # Prepare generation parameters
            parameters = {}
            if request.max_tokens:
                parameters["max_new_tokens"] = request.max_tokens
            if request.temperature is not None:
                parameters["temperature"] = request.temperature
            if request.top_p is not None:
                parameters["top_p"] = request.top_p
            if request.stop_sequences:
                parameters["stop_sequences"] = request.stop_sequences
            
            # For chat models, use chat completions
            if "instruct" in request.model_id.lower() or "chat" in request.model_id.lower():
                messages = [{"role": msg.role.value, "content": msg.content} for msg in request.messages]
                
                response = await self.client.chat_completion(
                    messages=messages,
                    model=request.model_id,
                    max_tokens=request.max_tokens or 1024,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    stop=request.stop_sequences
                )
                
                content = ""
                if response.choices and len(response.choices) > 0:
                    content = response.choices[0].message.content
                
                usage = {
                    'prompt_tokens': getattr(response.usage, 'prompt_tokens', 0),
                    'completion_tokens': getattr(response.usage, 'completion_tokens', 0),
                    'total_tokens': getattr(response.usage, 'total_tokens', 0),
                }
                
                finish_reason = response.choices[0].finish_reason if response.choices else None
                
            else:
                # For base models, use text generation
                response = await self.client.text_generation(
                    prompt=prompt,
                    model=request.model_id,
                    **parameters
                )
                
                content = response.generated_text
                
                # HF doesn't always provide usage info for text generation
                usage = {
                    'prompt_tokens': len(prompt.split()),  # rough estimate
                    'completion_tokens': len(content.split()),  # rough estimate
                    'total_tokens': len(prompt.split()) + len(content.split()),
                }
                
                finish_reason = getattr(response, 'finish_reason', None)
            
            return GenerationResponse(
                content=content,
                model_id=request.model_id,
                usage=usage,
                finish_reason=finish_reason,
                metadata={'provider': self.provider_name}
            )
            
        except Exception as e:
            raise RuntimeError(f"Hugging Face generation failed: {str(e)}")
    
    async def generate_stream(self, request: GenerationRequest) -> AsyncIterator[str]:
        """Generate streaming text using Hugging Face."""
        if self.client is None:
            await self.initialize()
        
        try:
            # Convert messages to prompt format
            prompt = self._convert_messages_to_prompt(request.messages)
            
            # Prepare generation parameters
            parameters = {}
            if request.max_tokens:
                parameters["max_new_tokens"] = request.max_tokens
            if request.temperature is not None:
                parameters["temperature"] = request.temperature
            if request.top_p is not None:
                parameters["top_p"] = request.top_p
            if request.stop_sequences:
                parameters["stop_sequences"] = request.stop_sequences
            
            # Use text generation stream
            async for token in self.client.text_generation(
                prompt=prompt,
                model=request.model_id,
                stream=True,
                **parameters
            ):
                if hasattr(token, 'token'):
                    yield token.token.text
                else:
                    yield str(token)
                    
        except Exception as e:
            raise RuntimeError(f"Hugging Face streaming failed: {str(e)}")
    
    def _convert_messages_to_prompt(self, messages: List[ChatMessage]) -> str:
        """Convert messages to a prompt format suitable for HF models."""
        prompt_parts = []
        
        for message in messages:
            if message.role == MessageRole.SYSTEM:
                prompt_parts.append(f"System: {message.content}")
            elif message.role == MessageRole.USER:
                prompt_parts.append(f"User: {message.content}")
            elif message.role == MessageRole.ASSISTANT:
                prompt_parts.append(f"Assistant: {message.content}")
        
        # Add assistant prompt at the end
        prompt_parts.append("Assistant:")
        
        return "\n".join(prompt_parts)
    
    async def health_check(self) -> bool:
        """Check if Hugging Face is accessible."""
        try:
            if self.client is None:
                await self.initialize()
            
            # Try a simple generation with a popular model
            test_response = await self.client.text_generation(
                prompt="Hello",
                model="microsoft/DialoGPT-medium",
                max_new_tokens=5
            )
            
            return test_response is not None
            
        except Exception:
            return False