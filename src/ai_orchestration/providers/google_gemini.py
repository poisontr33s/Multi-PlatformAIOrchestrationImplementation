"""
Google Gemini Provider for AI Orchestration
Uses the official google-generativeai SDK for AI Studio integration.
"""

import os
from typing import Any, Dict, List, Optional

import google.generativeai as genai
import structlog
from pydantic import BaseModel

logger = structlog.get_logger(__name__)


class GenerateRequest(BaseModel):
    """Request model for text generation."""

    prompt: str
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None


class GenerateResponse(BaseModel):
    """Response model for text generation."""

    text: str
    model: str
    usage: Dict[str, Any] = {}


class ModelInfo(BaseModel):
    """Model information."""

    name: str
    display_name: str
    description: str = ""
    input_token_limit: Optional[int] = None
    output_token_limit: Optional[int] = None


class GoogleGeminiProvider:
    """
    Google Gemini provider using the google-generativeai SDK.

    Supports AI Studio with GOOGLE_API_KEY and is designed to be extended
    for Vertex AI in future iterations.
    """

    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None):
        """
        Initialize Google Gemini provider.

        Args:
            api_key: Google API key (defaults to GOOGLE_API_KEY env var)
            model_name: Model to use (defaults to GOOGLE_GENAI_MODEL env var)
        """
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Google API key is required. Set GOOGLE_API_KEY environment variable "
                "or pass api_key parameter.",
            )

        # Configure the API
        genai.configure(api_key=self.api_key)

        # Set model with environment-driven configuration
        # TODO: Set GOOGLE_GENAI_MODEL explicitly in production
        self.model_name = model_name or os.getenv(
            "GOOGLE_GENAI_MODEL",
            "gemini-1.5-flash",  # Safe fallback to current non-deprecated model
        )

        self.model = None
        logger.info("GoogleGeminiProvider initialized", model=self.model_name)

    async def initialize(self) -> None:
        """Initialize the provider and validate model availability."""
        try:
            self.model = genai.GenerativeModel(self.model_name)
            # Test with a simple prompt to validate configuration
            test_response = self.model.generate_content("Hello")
            logger.info("Provider initialized successfully", model=self.model_name)
        except Exception as e:
            logger.error(
                "Failed to initialize provider", error=str(e), model=self.model_name
            )
            raise

    async def generate_text(self, request: GenerateRequest) -> GenerateResponse:
        """
        Generate text using Google Gemini.

        Args:
            request: Generation request with prompt and optional parameters

        Returns:
            GenerateResponse with generated text
        """
        if not self.model:
            raise RuntimeError("Provider not initialized. Call initialize() first.")

        try:
            # Configure generation parameters
            generation_config = {}
            if request.max_tokens:
                generation_config["max_output_tokens"] = request.max_tokens
            if request.temperature is not None:
                generation_config["temperature"] = request.temperature

            # Generate content
            response = self.model.generate_content(
                request.prompt,
                generation_config=(
                    genai.types.GenerationConfig(**generation_config)
                    if generation_config
                    else None
                ),
            )

            # Extract usage information if available
            usage_info = {}
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                usage_info = {
                    "prompt_tokens": getattr(
                        response.usage_metadata, "prompt_token_count", 0
                    ),
                    "completion_tokens": getattr(
                        response.usage_metadata, "candidates_token_count", 0
                    ),
                    "total_tokens": getattr(
                        response.usage_metadata, "total_token_count", 0
                    ),
                }

            return GenerateResponse(
                text=response.text,
                model=self.model_name,
                usage=usage_info,
            )

        except Exception as e:
            logger.error(
                "Text generation failed", error=str(e), prompt=request.prompt[:100]
            )
            raise

    async def list_models(self) -> List[ModelInfo]:
        """
        List available models.

        Returns:
            List of available model information
        """
        try:
            models = []
            for model in genai.list_models():
                # Filter for generative models
                if "generateContent" in model.supported_generation_methods:
                    models.append(
                        ModelInfo(
                            name=model.name,
                            display_name=model.display_name,
                            description=getattr(model, "description", ""),
                            input_token_limit=getattr(model, "input_token_limit", None),
                            output_token_limit=getattr(
                                model, "output_token_limit", None
                            ),
                        )
                    )

            logger.info("Listed models", count=len(models))
            return models

        except Exception as e:
            logger.error("Failed to list models", error=str(e))
            raise

    async def health_check(self) -> Dict[str, Any]:
        """
        Check provider health and configuration.

        Returns:
            Health status information
        """
        try:
            # Test basic functionality
            if self.model:
                test_response = self.model.generate_content("test")

            return {
                "status": "healthy",
                "provider": "google_gemini",
                "model": self.model_name,
                "api_configured": bool(self.api_key),
                "model_initialized": bool(self.model),
            }

        except Exception as e:
            return {
                "status": "unhealthy",
                "provider": "google_gemini",
                "model": self.model_name,
                "error": str(e),
                "api_configured": bool(self.api_key),
                "model_initialized": bool(self.model),
            }
