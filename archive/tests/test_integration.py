"""
Basic integration test for the AI Orchestration system.

Tests the core functionality and integration between components.
"""

import asyncio
import pytest
from datetime import datetime

from ai_orchestration import (
    Orchestrator,
    OrchestrationConfig,
    OrchestrationTask,
    TaskClassification,
    PriorityLevel,
    ComplexityRating,
    SubscriptionManager,
    SubscriptionTier,
    AgentCoordinator,
    AgentType,
    AgentCapability,
    EmergentIntelligence,
    EmergentConfig
)


@pytest.mark.asyncio
async def test_orchestrator_initialization():
    """Test orchestrator initialization and basic functionality."""
    config = OrchestrationConfig(
        max_concurrent_tasks=5,
        task_timeout_seconds=60
    )
    
    orchestrator = Orchestrator(config)
    await orchestrator.initialize()
    
    # Test task submission
    task = OrchestrationTask(
        id="test_task_001",
        classification=TaskClassification.AUTONOMOUS_STRATEGIC_IMPLEMENTATION,
        priority=PriorityLevel.CRITICAL_INFRASTRUCTURE,
        complexity=ComplexityRating.ARCHITECTURAL_SYNTHESIS,
        payload={"test": "data"}
    )
    
    task_id = await orchestrator.submit_task(task)
    assert task_id == "test_task_001"
    
    # Check metrics
    metrics = orchestrator.get_metrics()
    assert metrics["total_tasks"] >= 1
    
    await orchestrator.shutdown()


@pytest.mark.asyncio 
async def test_subscription_manager():
    """Test subscription management functionality."""
    manager = SubscriptionManager()
    await manager.initialize()
    
    # Register a user
    subscription = await manager.register_user(
        "user_001", 
        SubscriptionTier.PREMIUM,
        duration_days=30
    )
    
    assert subscription.user_id == "user_001"
    assert subscription.plan.tier == SubscriptionTier.PREMIUM
    
    # Check feature access
    from ai_orchestration.subscription.manager import SubscriptionFeature
    has_access = manager.check_feature_access("user_001", SubscriptionFeature.STRATEGIC_INTELLIGENCE)
    assert has_access is True
    
    # Check metrics
    metrics = manager.get_subscription_metrics()
    assert metrics["total_users"] >= 1
    
    await manager.shutdown()


@pytest.mark.asyncio
async def test_agent_coordinator():
    """Test agent coordination functionality."""
    coordinator = AgentCoordinator()
    await coordinator.initialize()
    
    # Deploy an agent
    agent_id = await coordinator.deploy_agent(
        AgentType.STRATEGIC_ADVISOR,
        [AgentCapability.DECISION_MAKING, AgentCapability.ANALYSIS],
        "premium"
    )
    
    assert agent_id is not None
    
    # Assign a task
    assigned_agent = await coordinator.assign_task(
        "test_task_001",
        [AgentCapability.DECISION_MAKING],
        priority=0.8,
        subscription_tier="premium"
    )
    
    assert assigned_agent is not None
    
    # Complete the task
    await coordinator.complete_task(assigned_agent, "test_task_001", success=True)
    
    # Check metrics
    metrics = coordinator.get_coordination_metrics()
    assert metrics["total_agents"] >= 1
    
    await coordinator.shutdown()


@pytest.mark.asyncio
async def test_emergent_intelligence():
    """Test emergent intelligence functionality."""
    config = EmergentConfig(
        enable_pattern_discovery=True,
        enable_adaptive_learning=True,
        min_pattern_confidence=0.5
    )
    
    intelligence = EmergentIntelligence(config)
    await intelligence.initialize()
    
    # Feed some test data
    test_data = [
        {"user_id": "user1", "action": "login", "performance": 0.8, "timestamp": datetime.utcnow().isoformat()},
        {"user_id": "user1", "action": "task_execute", "performance": 0.9, "timestamp": datetime.utcnow().isoformat()},
        {"user_id": "user2", "action": "login", "performance": 0.7, "timestamp": datetime.utcnow().isoformat()},
        {"user_id": "user2", "action": "task_execute", "performance": 0.85, "timestamp": datetime.utcnow().isoformat()},
    ]
    
    for data in test_data:
        await intelligence.feed_data(data)
    
    # Try to discover patterns
    patterns = await intelligence.discover_patterns()
    
    # Check metrics
    metrics = intelligence.get_intelligence_metrics()
    assert isinstance(metrics.patterns_discovered, int)
    
    await intelligence.shutdown()


@pytest.mark.asyncio
async def test_system_integration():
    """Test integration between all major components."""
    
    # Initialize all components
    orch_config = OrchestrationConfig(max_concurrent_tasks=10)
    orchestrator = Orchestrator(orch_config)
    await orchestrator.initialize()
    
    subscription_manager = SubscriptionManager()
    await subscription_manager.initialize()
    
    agent_coordinator = AgentCoordinator()
    await agent_coordinator.initialize()
    
    intel_config = EmergentConfig()
    intelligence = EmergentIntelligence(intel_config)
    await intelligence.initialize()
    
    try:
        # Register a premium user
        await subscription_manager.register_user("premium_user", SubscriptionTier.PREMIUM)
        
        # Submit a strategic task
        task = OrchestrationTask(
            id="integration_test_001",
            classification=TaskClassification.AUTONOMOUS_STRATEGIC_IMPLEMENTATION,
            priority=PriorityLevel.CRITICAL_INFRASTRUCTURE,
            complexity=ComplexityRating.ARCHITECTURAL_SYNTHESIS,
            payload={"integration_test": True}
        )
        
        task_id = await orchestrator.submit_task(task)
        
        # Deploy an agent for the task
        agent_id = await agent_coordinator.deploy_agent(
            AgentType.STRATEGIC_ADVISOR,
            [AgentCapability.DECISION_MAKING, AgentCapability.OPTIMIZATION],
            "premium"
        )
        
        # Assign the task to an agent
        assigned_agent = await agent_coordinator.assign_task(
            task_id,
            [AgentCapability.DECISION_MAKING],
            priority=1.0,
            subscription_tier="premium"
        )
        
        # Feed performance data to intelligence system
        await intelligence.feed_data({
            "task_id": task_id,
            "agent_id": assigned_agent,
            "performance": 0.95,
            "execution_time": 1.2,
            "priority": "critical"
        })
        
        # Complete the task
        await agent_coordinator.complete_task(assigned_agent, task_id, success=True)
        
        # Verify system state
        orch_metrics = orchestrator.get_metrics()
        agent_metrics = agent_coordinator.get_coordination_metrics()
        sub_metrics = subscription_manager.get_subscription_metrics()
        intel_metrics = intelligence.get_intelligence_metrics()
        
        assert orch_metrics["total_tasks"] >= 1
        assert agent_metrics["total_agents"] >= 1
        assert sub_metrics["total_users"] >= 1
        assert isinstance(intel_metrics.patterns_discovered, int)
        
    finally:
        # Cleanup
        await intelligence.shutdown()
        await agent_coordinator.shutdown()
        await subscription_manager.shutdown()
        await orchestrator.shutdown()


if __name__ == "__main__":
    # Run a simple integration test
    print("Running AI Orchestration Integration Test...")
    asyncio.run(test_system_integration())
    print("Integration test completed successfully!")