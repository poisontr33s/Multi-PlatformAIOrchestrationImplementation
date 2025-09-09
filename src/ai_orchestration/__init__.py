"""
Multi-Platform AI Orchestration Implementation

A comprehensive orchestration paradigm that systematically exploits the emergent
intelligence potential inherent in users' premium subscription matrix.

Strategic Task Specification:
- Task Classification: AUTONOMOUS_STRATEGIC_IMPLEMENTATION
- Priority Level: CRITICAL_INFRASTRUCTURE  
- Complexity Rating: ARCHITECTURAL_SYNTHESIS
- Execution Mode: DISTRIBUTED_AUTONOMOUS_ORCHESTRATION
"""

__version__ = "1.0.0"
__author__ = "AI Orchestration Team"
__email__ = "team@ai-orchestration.com"

from .core.orchestrator import (
    Orchestrator,
    OrchestrationTask,
    OrchestrationConfig,
    TaskClassification,
    PriorityLevel,
    ComplexityRating
)
from .core.engine import (
    StrategicEngine,
    StrategicDecision,
    StrategicDecisionType,
    StrategicContext
)
from .subscription.manager import (
    SubscriptionManager,
    SubscriptionTier,
    SubscriptionFeature,
    SubscriptionPlan,
    UserSubscription,
    SubscriptionMatrix
)
from .agents.coordinator import (
    AgentCoordinator,
    Agent,
    AgentType,
    AgentStatus,
    AgentCapability,
    AgentMetrics,
    CoordinationStrategy
)
from .intelligence.emergent import (
    EmergentIntelligence,
    LearningPattern,
    IntelligenceMetrics,
    EmergentConfig,
    LearningMode,
    PatternType
)

__all__ = [
    # Core orchestration
    "Orchestrator",
    "OrchestrationTask",
    "OrchestrationConfig",
    "TaskClassification",
    "PriorityLevel",
    "ComplexityRating",
    
    # Strategic engine
    "StrategicEngine",
    "StrategicDecision",
    "StrategicDecisionType",
    "StrategicContext",
    
    # Subscription management
    "SubscriptionManager",
    "SubscriptionTier",
    "SubscriptionFeature",
    "SubscriptionPlan",
    "UserSubscription",
    "SubscriptionMatrix",
    
    # Agent coordination
    "AgentCoordinator",
    "Agent",
    "AgentType", 
    "AgentStatus",
    "AgentCapability",
    "AgentMetrics",
    "CoordinationStrategy",
    
    # Emergent intelligence
    "EmergentIntelligence",
    "LearningPattern",
    "IntelligenceMetrics",
    "EmergentConfig",
    "LearningMode",
    "PatternType"
]