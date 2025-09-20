"""
Google Gemini Provider for AI Orchestration (2025 Standards)
Uses the official google-generativeai SDK for AI Studio integration.

Supports the latest Gemini 2.0 and 2.5 models with enhanced capabilities:
- 1M+ token context windows
- Advanced reasoning with thinking variants
- Multimodal generation capabilities
- Optimized for September 2025 AI orchestration patterns
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
    Google Gemini provider using the google-generativeai SDK (2025 Standards).

    Supports AI Studio with GOOGLE_API_KEY and is designed to be extended
    for Vertex AI in future iterations.
    
    2025 Model Support:
    - Gemini 2.0 Flash Experimental: Recommended default, 1M+ token context
    - Gemini 2.0 Flash Thinking: Advanced reasoning capabilities
    - Gemini 2.5 Preview models: Latest features and improvements
    - Full backwards compatibility with Gemini 1.5 series
    """

    # 2025 Model Registry with current capabilities
    SUPPORTED_MODELS = {
        # Gemini 2.0 Series (Recommended for 2025)
        "gemini-2.0-flash-exp": {
            "display_name": "Gemini 2.0 Flash Experimental", 
            "context_window": 1048576,
            "recommended": True,
            "features": ["text", "multimodal", "fast"]
        },
        "gemini-2.0-flash-thinking-exp": {
            "display_name": "Gemini 2.0 Flash Thinking Experimental",
            "context_window": 1048576, 
            "recommended": True,
            "features": ["text", "reasoning", "thinking"]
        },
        "gemini-2.0-flash": {
            "display_name": "Gemini 2.0 Flash",
            "context_window": 1048576,
            "recommended": True,
            "features": ["text", "multimodal", "stable"]
        },
        # Gemini 2.5 Series (Preview)
        "gemini-2.5-flash-preview": {
            "display_name": "Gemini 2.5 Flash Preview",
            "context_window": 1048576,
            "recommended": False,
            "features": ["text", "multimodal", "preview"]
        },
        "gemini-2.5-pro-preview": {
            "display_name": "Gemini 2.5 Pro Preview", 
            "context_window": 1048576,
            "recommended": False,
            "features": ["text", "multimodal", "preview", "high-quality"]
        },
        # Gemini 1.5 Series (Backwards compatibility)
        "gemini-1.5-flash": {
            "display_name": "Gemini 1.5 Flash",
            "context_window": 1000000,
            "recommended": False,
            "features": ["text", "multimodal", "legacy"]
        },
        "gemini-1.5-pro": {
            "display_name": "Gemini 1.5 Pro",
            "context_window": 2000000,
            "recommended": False, 
            "features": ["text", "multimodal", "legacy", "high-quality"]
        },
    }

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
        # Updated for 2025: Gemini 2.0 Flash Experimental is now the recommended default
        self.model_name = model_name or os.getenv(
            "GOOGLE_GENAI_MODEL",
            "gemini-2.0-flash-exp",  # 2025 default: 1M+ token context, superior reasoning
        )

        # Validate model against registry
        if self.model_name not in self.SUPPORTED_MODELS:
            logger.warning(
                "Model not in registry, proceeding anyway", 
                model=self.model_name,
                supported_models=list(self.SUPPORTED_MODELS.keys())
            )

        self.model = None
        logger.info(
            "GoogleGeminiProvider initialized", 
            model=self.model_name,
            context_window=self.get_model_context_window(),
            features=self.get_model_features()
        )

    def get_model_context_window(self) -> int:
        """Get the context window size for the current model."""
        return self.SUPPORTED_MODELS.get(self.model_name, {}).get("context_window", 128000)
    
    def get_model_features(self) -> List[str]:
        """Get the features supported by the current model."""
        return self.SUPPORTED_MODELS.get(self.model_name, {}).get("features", ["text"])
    
    def is_model_recommended(self) -> bool:
        """Check if the current model is recommended for 2025 usage."""
        return self.SUPPORTED_MODELS.get(self.model_name, {}).get("recommended", False)

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
            Health status information with 2025 model capabilities
        """
        try:
            # Test basic functionality
            if self.model:
                test_response = self.model.generate_content("test")

            return {
                "status": "healthy",
                "provider": "google_gemini",
                "model": self.model_name,
                "model_info": {
                    "display_name": self.SUPPORTED_MODELS.get(self.model_name, {}).get("display_name", self.model_name),
                    "context_window": self.get_model_context_window(),
                    "features": self.get_model_features(),
                    "recommended_2025": self.is_model_recommended(),
                },
                "api_configured": bool(self.api_key),
                "model_initialized": bool(self.model),
                "2025_compliance": {
                    "using_modern_model": self.model_name.startswith("gemini-2."),
                    "large_context_support": self.get_model_context_window() >= 1000000,
                    "multimodal_capable": "multimodal" in self.get_model_features(),
                }
            }

        except Exception as e:
            return {
                "status": "unhealthy",
                "provider": "google_gemini",
                "model": self.model_name,
                "error": str(e),
                "api_configured": bool(self.api_key),
                "model_initialized": bool(self.model),
                "2025_compliance": {
                    "using_modern_model": False,
                    "large_context_support": False,
                    "multimodal_capable": False,
                }
            }
