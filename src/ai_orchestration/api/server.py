"""
FastAPI server for Multi-Platform AI Orchestration (2025 Standards)
Minimal API surface for Phase 1: Google Gemini 2.0+ with enhanced capabilities
"""

from typing import Any, Dict, List

import structlog
from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import JSONResponse

from ai_orchestration.providers.google_gemini import (
    GenerateRequest,
    GenerateResponse,
    GoogleGeminiProvider,
    ModelInfo,
)

logger = structlog.get_logger(__name__)

# FastAPI app instance
app = FastAPI(
    title="Multi-Platform AI Orchestration API (2025)",
    description="Strategic AI orchestration with provider abstraction - Phase 1: Google Gemini 2.0+ with 1M+ token context windows",
    version="1.1.0",  # Updated for 2025 features
    docs_url="/docs",
    redoc_url="/redoc",
)

# Global provider instance (initialized on startup)
google_provider: GoogleGeminiProvider = None


async def get_google_provider() -> GoogleGeminiProvider:
    """Dependency to get the Google Gemini provider instance."""
    if not google_provider:
        raise HTTPException(
            status_code=503,
            detail="Google Gemini provider not initialized",
        )
    return google_provider


@app.on_event("startup")
async def startup_event():
    """Initialize providers on startup."""
    global google_provider
    try:
        google_provider = GoogleGeminiProvider()
        await google_provider.initialize()
        logger.info("FastAPI server initialized with Google Gemini provider")
    except Exception as e:
        logger.error("Failed to initialize Google Gemini provider", error=str(e))
        # Continue startup even if provider fails - allows health checks


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("FastAPI server shutting down")


@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """
    Health check endpoint.
    Returns overall system health and provider status.
    """
    try:
        health_info = {
            "status": "healthy",
            "service": "ai-orchestration-api",
            "version": "1.0.0",
            "providers": {},
        }

        # Check Google provider health
        if google_provider:
            provider_health = await google_provider.health_check()
            health_info["providers"]["google_gemini"] = provider_health
        else:
            health_info["providers"]["google_gemini"] = {
                "status": "not_initialized",
                "provider": "google_gemini",
            }

        # Determine overall status
        provider_statuses = [
            p.get("status", "unknown") for p in health_info["providers"].values()
        ]
        if any(status == "unhealthy" for status in provider_statuses):
            health_info["status"] = "degraded"
        elif any(status == "not_initialized" for status in provider_statuses):
            health_info["status"] = "starting"

        return health_info

    except Exception as e:
        logger.error("Health check failed", error=str(e))
        return {
            "status": "unhealthy",
            "service": "ai-orchestration-api",
            "error": str(e),
        }


@app.get("/models", response_model=List[ModelInfo])
async def list_models(
    provider: GoogleGeminiProvider = Depends(get_google_provider),
) -> List[ModelInfo]:
    """
    List available models from Google Gemini provider.
    Phase 1: Google only, will be extended for multi-provider in future.
    """
    try:
        models = await provider.list_models()
        logger.info("Listed models", provider="google_gemini", count=len(models))
        return models

    except Exception as e:
        logger.error("Failed to list models", error=str(e), provider="google_gemini")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve models: {e!s}",
        )


@app.post("/generate", response_model=GenerateResponse)
async def generate_text(
    request: GenerateRequest,
    provider: GoogleGeminiProvider = Depends(get_google_provider),
) -> GenerateResponse:
    """
    Generate text using Google Gemini.

    Phase 1: Google only endpoint.
    Future phases will include provider selection and routing.
    """
    try:
        response = await provider.generate_text(request)
        logger.info(
            "Text generated successfully",
            provider="google_gemini",
            model=response.model,
            prompt_length=len(request.prompt),
            response_length=len(response.text),
        )
        return response

    except Exception as e:
        logger.error(
            "Text generation failed",
            error=str(e),
            provider="google_gemini",
            prompt=request.prompt[:100],  # Log first 100 chars for debugging
        )
        raise HTTPException(
            status_code=500,
            detail=f"Text generation failed: {e!s}",
        )


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for better error responses."""
    logger.error("Unhandled exception", error=str(exc), path=str(request.url))
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": "An unexpected error occurred",
        },
    )


# TODO: Add provider selection endpoint for multi-provider routing
# TODO: Add streaming support for real-time generation
# TODO: Add function calling support for Gemini function calling
# TODO: Add Vertex AI integration alongside AI Studio
# TODO: Add authentication and rate limiting
# TODO: Add request/response validation and sanitization
