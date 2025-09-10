"""
Strategic Engine for Autonomous Decision Making

Implements the core strategic decision-making capabilities for autonomous
operations with emergent intelligence potential.
"""

import asyncio
import json
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

import structlog
import numpy as np
from pydantic import BaseModel

from .orchestrator import OrchestrationTask, TaskClassification, PriorityLevel

logger = structlog.get_logger(__name__)


class StrategicDecisionType(Enum):
    """Types of strategic decisions the engine can make."""
    RESOURCE_ALLOCATION = "resource_allocation"
    TASK_PRIORITIZATION = "task_prioritization"
    AGENT_DEPLOYMENT = "agent_deployment"
    CAPACITY_SCALING = "capacity_scaling"
    EMERGENT_ADAPTATION = "emergent_adaptation"


class StrategicContext(BaseModel):
    """Context information for strategic decision making."""
    current_load: float
    available_resources: Dict[str, int]
    historical_performance: List[float]
    subscription_metrics: Dict[str, Any]
    system_health: float
    prediction_horizon: int = 300  # seconds


@dataclass
class StrategicDecision:
    """Represents a strategic decision made by the engine."""
    decision_type: StrategicDecisionType
    parameters: Dict[str, Any]
    confidence: float
    reasoning: str
    timestamp: datetime
    expected_impact: Dict[str, float]


class StrategicEngine:
    """
    Autonomous strategic decision-making engine.
    
    Provides intelligent resource allocation, task prioritization, and adaptive
    system optimization based on real-time analytics and historical patterns.
    """
    
    def __init__(self):
        self.logger = structlog.get_logger(self.__class__.__name__)
        self.decision_history: List[StrategicDecision] = []
        self.learning_weights = np.array([0.4, 0.3, 0.2, 0.1])  # [performance, load, subscription, health]
        self._running = False
        
    async def initialize(self) -> None:
        """Initialize the strategic engine."""
        self.logger.info("Initializing StrategicEngine")
        self._running = True
        
        # Start background optimization loop
        asyncio.create_task(self._optimization_loop())
        
    async def shutdown(self) -> None:
        """Shutdown the strategic engine."""
        self.logger.info("Shutting down StrategicEngine")
        self._running = False
        
    async def make_strategic_decision(
        self, 
        context: StrategicContext,
        decision_type: StrategicDecisionType
    ) -> StrategicDecision:
        """
        Make a strategic decision based on current context.
        
        Args:
            context: Current system context
            decision_type: Type of decision to make
            
        Returns:
            Strategic decision with parameters and reasoning
        """
        self.logger.info("Making strategic decision", decision_type=decision_type.value)
        
        if decision_type == StrategicDecisionType.RESOURCE_ALLOCATION:
            decision = await self._decide_resource_allocation(context)
        elif decision_type == StrategicDecisionType.TASK_PRIORITIZATION:
            decision = await self._decide_task_prioritization(context)
        elif decision_type == StrategicDecisionType.AGENT_DEPLOYMENT:
            decision = await self._decide_agent_deployment(context)
        elif decision_type == StrategicDecisionType.CAPACITY_SCALING:
            decision = await self._decide_capacity_scaling(context)
        elif decision_type == StrategicDecisionType.EMERGENT_ADAPTATION:
            decision = await self._decide_emergent_adaptation(context)
        else:
            raise ValueError(f"Unknown decision type: {decision_type}")
            
        self.decision_history.append(decision)
        
        # Keep only recent decisions for memory management
        if len(self.decision_history) > 1000:
            self.decision_history = self.decision_history[-800:]
            
        return decision
        
    async def _decide_resource_allocation(self, context: StrategicContext) -> StrategicDecision:
        """Decide on optimal resource allocation."""
        # Calculate resource allocation based on load and performance
        total_cpu = context.available_resources.get("cpu", 100)
        total_memory = context.available_resources.get("memory", 100)
        
        # Strategic allocation based on subscription tiers and performance
        premium_allocation = min(0.7, context.current_load + 0.2)
        standard_allocation = min(0.5, context.current_load)
        
        allocation = {
            "premium_cpu_percent": premium_allocation,
            "premium_memory_percent": premium_allocation,
            "standard_cpu_percent": standard_allocation,
            "standard_memory_percent": standard_allocation,
        }
        
        confidence = self._calculate_confidence(context, "resource_allocation")
        
        return StrategicDecision(
            decision_type=StrategicDecisionType.RESOURCE_ALLOCATION,
            parameters=allocation,
            confidence=confidence,
            reasoning=f"Allocated {premium_allocation:.2%} to premium, {standard_allocation:.2%} to standard based on current load {context.current_load:.2%}",
            timestamp=datetime.utcnow(),
            expected_impact={"performance_improvement": 0.15, "cost_efficiency": 0.10}
        )
        
    async def _decide_task_prioritization(self, context: StrategicContext) -> StrategicDecision:
        """Decide on task prioritization strategy."""
        # Priority weights based on subscription tier and system health
        priority_weights = {
            "critical_infrastructure": 1.0,
            "premium_tasks": 0.8 if context.system_health > 0.7 else 0.9,
            "standard_tasks": 0.5 if context.system_health > 0.5 else 0.3,
            "background_tasks": 0.2 if context.system_health > 0.8 else 0.1
        }
        
        confidence = self._calculate_confidence(context, "task_prioritization")
        
        return StrategicDecision(
            decision_type=StrategicDecisionType.TASK_PRIORITIZATION,
            parameters=priority_weights,
            confidence=confidence,
            reasoning=f"Prioritization adjusted for system health {context.system_health:.2%}",
            timestamp=datetime.utcnow(),
            expected_impact={"throughput_improvement": 0.12, "user_satisfaction": 0.18}
        )
        
    async def _decide_agent_deployment(self, context: StrategicContext) -> StrategicDecision:
        """Decide on agent deployment strategy."""
        # Calculate optimal agent count based on load and subscription metrics
        base_agents = 3
        load_factor = context.current_load
        premium_users = context.subscription_metrics.get("premium_users", 0)
        
        optimal_agents = max(base_agents, int(base_agents + load_factor * 5 + premium_users * 0.1))
        
        deployment = {
            "total_agents": optimal_agents,
            "specialized_agents": max(1, optimal_agents // 3),
            "general_agents": optimal_agents - max(1, optimal_agents // 3)
        }
        
        confidence = self._calculate_confidence(context, "agent_deployment")
        
        return StrategicDecision(
            decision_type=StrategicDecisionType.AGENT_DEPLOYMENT,
            parameters=deployment,
            confidence=confidence,
            reasoning=f"Deploying {optimal_agents} agents for load {load_factor:.2%} and {premium_users} premium users",
            timestamp=datetime.utcnow(),
            expected_impact={"response_time_improvement": 0.20, "scalability": 0.25}
        )
        
    async def _decide_capacity_scaling(self, context: StrategicContext) -> StrategicDecision:
        """Decide on capacity scaling strategy."""
        # Predictive scaling based on historical performance and current trends
        avg_performance = np.mean(context.historical_performance) if context.historical_performance else 0.5
        trend = self._calculate_performance_trend(context.historical_performance)
        
        if avg_performance < 0.6 or trend < -0.1:
            scale_action = "scale_up"
            scale_factor = 1.5
        elif avg_performance > 0.9 and trend > 0.1:
            scale_action = "scale_down"
            scale_factor = 0.8
        else:
            scale_action = "maintain"
            scale_factor = 1.0
            
        scaling = {
            "action": scale_action,
            "scale_factor": scale_factor,
            "target_capacity": int(100 * scale_factor)
        }
        
        confidence = self._calculate_confidence(context, "capacity_scaling")
        
        return StrategicDecision(
            decision_type=StrategicDecisionType.CAPACITY_SCALING,
            parameters=scaling,
            confidence=confidence,
            reasoning=f"Scaling {scale_action} with factor {scale_factor:.2f} based on performance {avg_performance:.2%} and trend {trend:.3f}",
            timestamp=datetime.utcnow(),
            expected_impact={"cost_optimization": 0.15, "performance_stability": 0.22}
        )
        
    async def _decide_emergent_adaptation(self, context: StrategicContext) -> StrategicDecision:
        """Decide on emergent intelligence adaptations."""
        # Emergent behavior based on pattern recognition and anomaly detection
        anomaly_score = self._detect_anomalies(context)
        adaptation_needed = anomaly_score > 0.3
        
        if adaptation_needed:
            adaptation = {
                "enable_advanced_learning": True,
                "adjust_decision_weights": True,
                "experimental_features": True,
                "anomaly_response_mode": "adaptive"
            }
            reasoning = f"Emergent adaptation triggered by anomaly score {anomaly_score:.3f}"
        else:
            adaptation = {
                "enable_advanced_learning": False,
                "adjust_decision_weights": False,
                "experimental_features": False,
                "anomaly_response_mode": "standard"
            }
            reasoning = "Standard operation mode, no emergent adaptation needed"
            
        confidence = self._calculate_confidence(context, "emergent_adaptation")
        
        return StrategicDecision(
            decision_type=StrategicDecisionType.EMERGENT_ADAPTATION,
            parameters=adaptation,
            confidence=confidence,
            reasoning=reasoning,
            timestamp=datetime.utcnow(),
            expected_impact={"adaptability": 0.30, "innovation_potential": 0.25}
        )
        
    def _calculate_confidence(self, context: StrategicContext, decision_type: str) -> float:
        """Calculate confidence score for a decision."""
        # Confidence based on data quality and historical success
        data_quality = min(1.0, len(context.historical_performance) / 10)
        system_stability = context.system_health
        historical_success = self._get_historical_success_rate(decision_type)
        
        confidence = (data_quality * 0.3 + system_stability * 0.4 + historical_success * 0.3)
        return max(0.1, min(1.0, confidence))
        
    def _calculate_performance_trend(self, performance_history: List[float]) -> float:
        """Calculate performance trend from historical data."""
        if len(performance_history) < 2:
            return 0.0
            
        recent = performance_history[-5:] if len(performance_history) >= 5 else performance_history
        if len(recent) < 2:
            return 0.0
            
        x = np.arange(len(recent))
        slope, _ = np.polyfit(x, recent, 1)
        return slope
        
    def _detect_anomalies(self, context: StrategicContext) -> float:
        """Detect system anomalies and return anomaly score."""
        if not context.historical_performance:
            return 0.0
            
        current_perf = context.system_health
        historical_mean = np.mean(context.historical_performance)
        historical_std = np.std(context.historical_performance) if len(context.historical_performance) > 1 else 0.1
        
        # Z-score based anomaly detection
        z_score = abs(current_perf - historical_mean) / historical_std if historical_std > 0 else 0
        anomaly_score = min(1.0, z_score / 3.0)  # Normalize to 0-1 range
        
        return anomaly_score
        
    def _get_historical_success_rate(self, decision_type: str) -> float:
        """Get historical success rate for a decision type."""
        relevant_decisions = [
            d for d in self.decision_history 
            if d.decision_type.value == decision_type
        ]
        
        if not relevant_decisions:
            return 0.7  # Default moderate confidence
            
        # Simplified success rate based on confidence scores
        avg_confidence = np.mean([d.confidence for d in relevant_decisions])
        return avg_confidence
        
    async def _optimization_loop(self) -> None:
        """Background optimization loop for continuous improvement."""
        while self._running:
            try:
                # Periodic learning and weight adjustment
                await self._update_learning_weights()
                await asyncio.sleep(60)  # Run every minute
            except Exception as e:
                self.logger.error("Error in optimization loop", error=str(e))
                await asyncio.sleep(10)
                
    async def _update_learning_weights(self) -> None:
        """Update learning weights based on recent performance."""
        if len(self.decision_history) < 10:
            return
            
        # Analyze recent decision outcomes and adjust weights
        recent_decisions = self.decision_history[-10:]
        avg_confidence = np.mean([d.confidence for d in recent_decisions])
        
        # Adjust weights based on performance
        if avg_confidence > 0.8:
            self.learning_weights = self.learning_weights * 1.01  # Slight increase
        elif avg_confidence < 0.6:
            self.learning_weights = self.learning_weights * 0.99  # Slight decrease
            
        # Normalize weights
        self.learning_weights = self.learning_weights / np.sum(self.learning_weights)
        
        self.logger.debug("Updated learning weights", weights=self.learning_weights.tolist())