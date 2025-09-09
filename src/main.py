"""
Main entry point for the Multi-Platform AI Orchestration System.
"""

import asyncio
import logging
import signal
import sys
from pathlib import Path

import structlog
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .orchestration.core import AIOrchestrator, OrchestrationConfig, OrchestrationMode
from .api.routes import api_router
from .monitoring.performance import AdvancedPerformanceOrchestrator


# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.dev.ConsoleRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger("main")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Multi-Platform AI Orchestration System",
        description="Comprehensive AI orchestration across multiple platforms",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include API routes
    app.include_router(api_router, prefix="/api/v1")
    
    return app


class OrchestrationServer:
    """Main orchestration server that manages the AI orchestration system."""
    
    def __init__(self):
        self.app = create_app()
        self.orchestrator: Optional[AIOrchestrator] = None
        self.performance_monitor: Optional[AdvancedPerformanceOrchestrator] = None
        self._shutdown_event = asyncio.Event()
    
    async def initialize(self):
        """Initialize the orchestration system."""
        try:
            logger.info("Initializing Multi-Platform AI Orchestration System")
            
            # Create orchestration configuration
            config = OrchestrationConfig(
                mode=OrchestrationMode.FULL_AUTONOMOUS,
                max_concurrent_tasks=10,
                enable_monitoring=True,
                enable_gpu_optimization=True
            )
            
            # Initialize orchestrator
            self.orchestrator = AIOrchestrator(config)
            await self.orchestrator.initialize()
            
            # Initialize performance monitoring
            self.performance_monitor = AdvancedPerformanceOrchestrator()
            await self.performance_monitor.start()
            
            # Store in app state for API access
            self.app.state.orchestrator = self.orchestrator
            self.app.state.performance_monitor = self.performance_monitor
            
            logger.info("Multi-Platform AI Orchestration System initialized successfully")
            
        except Exception as e:
            logger.error("Failed to initialize orchestration system", error=str(e))
            raise
    
    async def shutdown(self):
        """Shutdown the orchestration system gracefully."""
        logger.info("Shutting down Multi-Platform AI Orchestration System")
        
        self._shutdown_event.set()
        
        if self.orchestrator:
            await self.orchestrator.shutdown()
        
        if self.performance_monitor:
            await self.performance_monitor.shutdown()
        
        logger.info("Multi-Platform AI Orchestration System shutdown complete")
    
    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating shutdown")
            asyncio.create_task(self.shutdown())
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)


async def main():
    """Main entry point."""
    try:
        # Create and initialize server
        server = OrchestrationServer()
        server.setup_signal_handlers()
        await server.initialize()
        
        # Run the server
        config = uvicorn.Config(
            app=server.app,
            host="0.0.0.0",
            port=8000,
            reload=False,
            log_level="info"
        )
        
        server_instance = uvicorn.Server(config)
        
        # Run server until shutdown
        await server_instance.serve()
        
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down")
    except Exception as e:
        logger.error("Critical error in main", error=str(e))
        sys.exit(1)
    finally:
        if 'server' in locals():
            await server.shutdown()


if __name__ == "__main__":
    asyncio.run(main())