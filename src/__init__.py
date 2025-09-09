"""
Multi-Platform AI Orchestration System

A comprehensive orchestration system for coordinating between GitHub Copilot Pro+,
Google AI Pro/Ultra, Microsoft AI Pro, Jules Asynchronous Coding Agent, Firebase Studio,
and local Gemma 3/GPT-OSS models.
"""

__version__ = "1.0.0"
__author__ = "AI Orchestration Team"
__license__ = "Apache-2.0"

from .orchestration.core import AIOrchestrator
from .orchestration.models import ModelCoordinator
from .agents.jules import JulesAgent
from .integrations.firebase import FirebaseStudioBridge
from .integrations.google import GoogleAIIntegration
from .integrations.microsoft import MicrosoftAIIntegration

__all__ = [
    "AIOrchestrator",
    "ModelCoordinator", 
    "JulesAgent",
    "FirebaseStudioBridge",
    "GoogleAIIntegration",
    "MicrosoftAIIntegration",
]