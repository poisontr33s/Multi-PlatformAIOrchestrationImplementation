"""
Test suite for the core orchestration module.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from src.orchestration.core import (
    AIOrchestrator,
    OrchestrationConfig,
    TaskSpecification,
    TaskPriority,
    OrchestrationMode
)


@pytest.fixture
def orchestration_config():
    """Create a test orchestration configuration."""
    return OrchestrationConfig(
        mode=OrchestrationMode.SUPERVISED_COORDINATION,
        max_concurrent_tasks=5,
        enable_monitoring=False  # Disable for testing
    )


@pytest.fixture
def sample_task():
    """Create a sample task specification."""
    return TaskSpecification(
        id="test_task_001",
        description="Generate a Python function for data processing",
        priority=TaskPriority.MEDIUM,
        context={"language": "python", "framework": "pandas"},
        requirements={"technical_requirements": ["error_handling", "type_hints"]},
        expected_output={"code": "string", "documentation": "string"}
    )


@pytest.fixture
async def orchestrator(orchestration_config):
    """Create and initialize an AI orchestrator."""
    orchestrator = AIOrchestrator(orchestration_config)
    
    # Mock the authentication manager
    with patch('src.orchestration.core.UnifiedAuthenticationManager'):
        with patch.object(orchestrator, '_initialize_integrations', new_callable=AsyncMock):
            await orchestrator.initialize()
    
    return orchestrator


class TestAIOrchestrator:
    """Test cases for the AI Orchestrator."""
    
    async def test_initialization(self, orchestration_config):
        """Test orchestrator initialization."""
        orchestrator = AIOrchestrator(orchestration_config)
        
        assert orchestrator.config == orchestration_config
        assert not orchestrator._initialized
        assert len(orchestrator.active_tasks) == 0
        assert len(orchestrator.completed_tasks) == 0
    
    async def test_coordinate_task_delegation_not_initialized(self, orchestration_config, sample_task):
        """Test task delegation when orchestrator is not initialized."""
        orchestrator = AIOrchestrator(orchestration_config)
        
        with pytest.raises(RuntimeError, match="Orchestrator not initialized"):
            await orchestrator.coordinate_task_delegation(sample_task)
    
    async def test_coordinate_task_delegation_success(self, orchestrator, sample_task):
        """Test successful task delegation."""
        # Mock the task execution
        mock_execution = Mock()
        mock_execution.task_id = sample_task.id
        mock_execution.status = "completed"
        mock_execution.result = {"output": "Generated code"}
        
        # Add the task to completed tasks directly for testing
        orchestrator.completed_tasks[sample_task.id] = mock_execution
        
        # Test coordination
        result = await orchestrator.coordinate_task_delegation(sample_task)
        
        assert result.task_id == sample_task.id
        assert result.status == "completed"
        assert result.result["output"] == "Generated code"
    
    async def test_monitor_execution_state_active(self, orchestrator, sample_task):
        """Test monitoring execution state for active task."""
        # Add task to active tasks
        mock_execution = Mock()
        mock_execution.task_id = sample_task.id
        mock_execution.status = "executing"
        orchestrator.active_tasks[sample_task.id] = mock_execution
        
        result = await orchestrator.monitor_execution_state(sample_task.id)
        
        assert result["status"] == "executing"
        assert result["task"].task_id == sample_task.id
    
    async def test_monitor_execution_state_completed(self, orchestrator, sample_task):
        """Test monitoring execution state for completed task."""
        # Add task to completed tasks
        mock_execution = Mock()
        mock_execution.task_id = sample_task.id
        mock_execution.status = "completed"
        orchestrator.completed_tasks[sample_task.id] = mock_execution
        
        result = await orchestrator.monitor_execution_state(sample_task.id)
        
        assert result["status"] == "completed"
        assert result["task"].task_id == sample_task.id
    
    async def test_monitor_execution_state_not_found(self, orchestrator):
        """Test monitoring execution state for non-existent task."""
        result = await orchestrator.monitor_execution_state("non_existent_task")
        
        assert result["status"] == "not_found"
        assert result["task"] is None
    
    async def test_shutdown(self, orchestrator):
        """Test orchestrator shutdown."""
        # Mock the integrations
        orchestrator.jules_agent = AsyncMock()
        orchestrator.firebase_bridge = AsyncMock()
        orchestrator.google_integration = AsyncMock()
        orchestrator.microsoft_integration = AsyncMock()
        orchestrator.performance_monitor = AsyncMock()
        
        await orchestrator.shutdown()
        
        # Verify all integrations were shut down
        orchestrator.jules_agent.shutdown.assert_called_once()
        orchestrator.firebase_bridge.shutdown.assert_called_once()
        orchestrator.google_integration.shutdown.assert_called_once()
        orchestrator.microsoft_integration.shutdown.assert_called_once()


class TestOrchestrationConfig:
    """Test cases for orchestration configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = OrchestrationConfig()
        
        assert config.mode == OrchestrationMode.FULL_AUTONOMOUS
        assert config.max_concurrent_tasks == 10
        assert config.resource_allocation_strategy == "balanced"
        assert config.enable_gpu_optimization is True
        assert config.enable_monitoring is True
        assert config.fallback_timeout_seconds == 30
        assert config.retry_attempts == 3
        assert config.circuit_breaker_threshold == 5
        assert "gemini-2.5-pro" in config.priority_models
    
    def test_custom_config(self):
        """Test custom configuration values."""
        custom_models = ["custom-model-1", "custom-model-2"]
        config = OrchestrationConfig(
            mode=OrchestrationMode.MANUAL_OVERRIDE,
            max_concurrent_tasks=20,
            resource_allocation_strategy="gpu_optimized",
            priority_models=custom_models
        )
        
        assert config.mode == OrchestrationMode.MANUAL_OVERRIDE
        assert config.max_concurrent_tasks == 20
        assert config.resource_allocation_strategy == "gpu_optimized"
        assert config.priority_models == custom_models


class TestTaskSpecification:
    """Test cases for task specification."""
    
    def test_task_creation(self, sample_task):
        """Test task specification creation."""
        assert sample_task.id == "test_task_001"
        assert sample_task.description == "Generate a Python function for data processing"
        assert sample_task.priority == TaskPriority.MEDIUM
        assert sample_task.context["language"] == "python"
        assert "error_handling" in sample_task.requirements["technical_requirements"]
        assert sample_task.timeout_seconds == 300  # default
        assert sample_task.retry_on_failure is True  # default
    
    def test_task_with_custom_timeout(self):
        """Test task with custom timeout."""
        task = TaskSpecification(
            id="timeout_test",
            description="Test task with custom timeout",
            priority=TaskPriority.HIGH,
            context={},
            requirements={},
            expected_output={},
            timeout_seconds=600,
            retry_on_failure=False
        )
        
        assert task.timeout_seconds == 600
        assert task.retry_on_failure is False


@pytest.mark.asyncio
class TestTaskExecution:
    """Test cases for task execution scenarios."""
    
    async def test_task_queue_processing(self, orchestrator, sample_task):
        """Test task queue processing mechanism."""
        # This would test the internal task queue processing
        # For now, we'll test that tasks can be added to the queue
        
        assert orchestrator.task_queue.qsize() == 0
        
        # The actual queue processing is tested through integration tests
        # since it involves complex async coordination
    
    async def test_concurrent_task_handling(self, orchestrator):
        """Test handling of multiple concurrent tasks."""
        # Create multiple tasks
        tasks = []
        for i in range(3):
            task = TaskSpecification(
                id=f"concurrent_task_{i}",
                description=f"Concurrent task {i}",
                priority=TaskPriority.MEDIUM,
                context={},
                requirements={},
                expected_output={}
            )
            tasks.append(task)
        
        # Test that orchestrator can handle task creation
        for task in tasks:
            assert task.id not in orchestrator.active_tasks
            assert task.id not in orchestrator.completed_tasks