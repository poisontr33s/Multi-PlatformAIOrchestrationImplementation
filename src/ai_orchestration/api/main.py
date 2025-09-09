"""
Main FastAPI application for AI Orchestration API.
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import JSONResponse
from typing import List, Optional, Dict, Any
import logging

from .models import (
    ModelInfo, ChatRequest, ChatResponse, GenerateRequest, GenerateResponse,
    HealthResponse, ErrorResponse
)
from .providers import get_provider_manager
from ..providers import ProviderManager, MessageRole, ChatMessage, GenerationRequest

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Multi-Platform AI Orchestration API",
    description="""
    Strategic Multi-Platform AI Orchestration Implementation with Distributed Autonomous Capabilities.
    
    This API provides unified access to multiple AI providers:
    - Google Gemini (via google-genai SDK)
    - Anthropic Claude
    - OpenAI GPT
    - Hugging Face OSS models
    
    ## Features
    - **Multi-provider support**: Use different AI providers seamlessly
    - **OpenAPI compliance**: Full OpenAPI 3.0 specification
    - **Streaming support**: Real-time response streaming
    - **Load balancing**: Intelligent provider selection
    - **Health monitoring**: Provider health checks
    """,
    version="1.0.0",
    openapi_tags=[
        {
            "name": "models",
            "description": "Operations with AI models",
        },
        {
            "name": "generation", 
            "description": "Text and chat generation",
        },
        {
            "name": "health",
            "description": "Health checks and monitoring",
        },
    ],
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Multi-Platform AI Orchestration API",
        "version": "1.0.0",
        "description": "Strategic AI orchestration with distributed autonomous capabilities",
        "docs": "/docs",
        "openapi": "/openapi.json"
    }


@app.get("/health", response_model=HealthResponse, tags=["health"])
async def health_check(provider_manager: ProviderManager = Depends(get_provider_manager)):
    """
    Health check endpoint.
    
    Returns the health status of all registered AI providers.
    """
    try:
        provider_health = await provider_manager.health_check_all()
        
        overall_status = "healthy" if all(provider_health.values()) else "degraded"
        if not provider_health:
            overall_status = "unhealthy"
        
        return HealthResponse(
            status=overall_status,
            providers=provider_health
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Health check failed")


@app.get("/models", response_model=List[ModelInfo], tags=["models"])
async def list_models(
    provider: Optional[str] = None,
    provider_manager: ProviderManager = Depends(get_provider_manager)
):
    """
    List available AI models.
    
    - **provider**: Optional filter by provider name
    
    Returns a list of available models from all or specified providers.
    """
    try:
        if provider:
            # Get models from specific provider
            provider_obj = provider_manager.get_provider(provider)
            if not provider_obj:
                raise HTTPException(status_code=404, detail=f"Provider '{provider}' not found")
            
            models = await provider_obj.list_models()
            model_infos = [ModelInfo.from_ai_model(model) for model in models]
            
        else:
            # Get models from all providers
            all_models = await provider_manager.list_all_models()
            model_infos = []
            
            for provider_name, models in all_models.items():
                for model in models:
                    model_infos.append(ModelInfo.from_ai_model(model))
        
        return model_infos
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to list models: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to list models")


@app.post("/generate", response_model=GenerateResponse, tags=["generation"])
async def generate_text(
    request: GenerateRequest,
    provider_manager: ProviderManager = Depends(get_provider_manager)
):
    """
    Generate text completion.
    
    Generate text using the specified model and provider.
    """
    try:
        # Convert to internal format
        messages = [
            ChatMessage(role=MessageRole.USER, content=request.prompt)
        ]
        
        # Add system message if provided
        if request.system:
            messages.insert(0, ChatMessage(role=MessageRole.SYSTEM, content=request.system))
        
        generation_request = GenerationRequest(
            messages=messages,
            model_id=request.model,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            stop_sequences=request.stop,
            stream=request.stream
        )
        
        # Generate response
        response = await provider_manager.generate(generation_request, request.provider)
        
        return GenerateResponse(
            text=response.content,
            model=response.model_id,
            usage=response.usage,
            finish_reason=response.finish_reason
        )
        
    except Exception as e:
        logger.error(f"Generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


@app.post("/chat", response_model=ChatResponse, tags=["generation"])
async def chat_completion(
    request: ChatRequest,
    provider_manager: ProviderManager = Depends(get_provider_manager)
):
    """
    Create chat completion.
    
    Generate a chat completion using the conversation history.
    """
    try:
        # Convert messages to internal format
        messages = []
        for msg in request.messages:
            role = MessageRole(msg.role.value)
            messages.append(ChatMessage(role=role, content=msg.content))
        
        generation_request = GenerationRequest(
            messages=messages,
            model_id=request.model,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            stop_sequences=request.stop,
            stream=request.stream
        )
        
        # Generate response
        response = await provider_manager.generate(generation_request, request.provider)
        
        return ChatResponse(
            message={
                "role": "assistant",
                "content": response.content
            },
            model=response.model_id,
            usage=response.usage,
            finish_reason=response.finish_reason
        )
        
    except Exception as e:
        logger.error(f"Chat completion failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Chat completion failed: {str(e)}")


@app.get("/providers", tags=["models"])
async def list_providers(provider_manager: ProviderManager = Depends(get_provider_manager)):
    """List all registered AI providers."""
    try:
        providers = provider_manager.list_providers()
        default_provider = provider_manager.default_provider
        
        return {
            "providers": providers,
            "default": default_provider,
            "count": len(providers)
        }
        
    except Exception as e:
        logger.error(f"Failed to list providers: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to list providers")


# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content=ErrorResponse(
            error="Not Found",
            message="The requested resource was not found",
            code=404
        ).dict()
    )


@app.exception_handler(500)
async def internal_server_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal Server Error",
            message="An unexpected error occurred",
            code=500
        ).dict()
    )