# Jules Asynchronous Agent Documentation

Jules is the asynchronous agent system for the Multi-Platform AI Orchestration Implementation. This document provides guidance for agent configuration, execution, and file block emission.

## Agent Configuration

### Environment Setup

Jules agents operate within the uv-based Python environment. Use the setup script for quick initialization:

```bash
# Quick setup with linting and testing
./scripts/jules-setup.sh

# Manual setup
./scripts/dev-setup.sh
uv run ruff check . --fix
uv run black .
uv run pytest -q
```

### Agent Runtime Configuration

Jules agents are configured through environment variables and runtime parameters:

```bash
# Core configuration
AI_ORCHESTRATION_LOG_LEVEL=INFO
AI_ORCHESTRATION_MAX_TASKS=100
AI_ORCHESTRATION_ENABLE_EMERGENT=true

# Agent-specific configuration
AI_ORCHESTRATION_INITIAL_AGENTS=4
AI_ORCHESTRATION_MAX_AGENTS=50
AI_ORCHESTRATION_COORDINATION_STRATEGY=emergent_adaptive

# Intelligence configuration
AI_ORCHESTRATION_PATTERN_CONFIDENCE=0.7
AI_ORCHESTRATION_LEARNING_RETENTION_DAYS=30
AI_ORCHESTRATION_MODEL_UPDATE_FREQUENCY=60
```

## Snapshot and Configuration

### System Snapshot

Jules agents can take snapshots of the current system state for analysis and decision-making:

```python
from ai_orchestration import Orchestrator, AgentCoordinator, EmergentIntelligence

async def take_system_snapshot():
    """Take a comprehensive system snapshot."""
    snapshot = {
        "orchestrator_metrics": orchestrator.get_metrics(),
        "agent_metrics": agent_coordinator.get_coordination_metrics(),
        "intelligence_metrics": intelligence.get_intelligence_metrics(),
        "active_tasks": await orchestrator.get_active_tasks(),
        "provider_health": await check_provider_health(),
        "timestamp": datetime.utcnow().isoformat()
    }
    return snapshot
```

### Configuration Runtime

Agents can modify their configuration at runtime based on system conditions:

```python
async def adaptive_configuration(current_load: float, error_rate: float):
    """Adapt agent configuration based on system metrics."""
    if current_load > 0.8:
        # Increase agent count for high load
        await agent_coordinator.scale_agents(target_count=min(50, current_agents * 1.5))
    
    if error_rate > 0.1:
        # Reduce complexity for high error rates
        await orchestrator.update_config(max_concurrent_tasks=max(5, current_tasks * 0.8))
```

## File Block Emission

### Agent File Block Format

Jules agents must emit file blocks in the following format when generating or modifying code:

#### Python Files

```python name=src/path/to/file.py
"""
Module docstring
"""

import os
from typing import Dict, Any

def example_function() -> Dict[str, Any]:
    """Example function with proper formatting."""
    return {"status": "success"}

# Rest of the Python code
```

#### Markdown Files

````markdown name=docs/EXAMPLE.md
# Example Documentation

This is an example of how to format markdown file blocks.

## Section

Content here.

```bash
# Example command
uv run pytest -q
```
````

#### YAML/Configuration Files

```yaml name=config/example.yaml
# Example configuration
version: "1.0"
settings:
  enabled: true
  max_retries: 3
```

#### JSON Files

```json name=config/example.json
{
  "version": "1.0.0",
  "configuration": {
    "enabled": true,
    "providers": ["google_gemini"]
  }
}
```

### Applying File Blocks

Use the file block application script to apply agent-generated changes:

```bash
# Apply file blocks from agent output
python scripts/apply_file_blocks.py < agent_output.md

# Or pipe directly from agent
jules-agent generate-code | python scripts/apply_file_blocks.py
```

### File Block Best Practices

1. **Always include the full file path** relative to the project root
2. **Use appropriate language tags** for syntax highlighting
3. **Include proper imports and structure** for Python files
4. **Add docstrings and comments** following project conventions
5. **Ensure code follows linting rules** (ruff, black)

## Integration with Provider System

### Google Gemini Integration

Jules agents can interact with the Google Gemini provider:

```python
from ai_orchestration.providers.google_gemini import GoogleGeminiProvider

async def agent_gemini_interaction():
    """Example of agent using Gemini provider."""
    provider = GoogleGeminiProvider()
    await provider.initialize()
    
    # Generate strategic analysis
    response = await provider.generate_text(GenerateRequest(
        prompt="Analyze the current system metrics and suggest optimizations",
        temperature=0.3
    ))
    
    return response.text
```

### Node CLI Bridge Usage

Agents can use the Node CLI bridge for JavaScript/TypeScript tooling:

```python
from ai_orchestration.bridge.node_cli import NodeCLIBridge

async def agent_node_tooling():
    """Example of agent using Node CLI bridge."""
    bridge = NodeCLIBridge()
    
    # Future: Execute gemini CLI (placeholder)
    result = await bridge.gemini_cli_placeholder(["generate", "--prompt", "Hello"])
    
    # Execute other Node tools
    typescript_check = await bridge.execute_command(
        "tsc", 
        ["--noEmit", "--project", "tsconfig.json"]
    )
    
    return result
```

## Agent Development Guidelines

### Code Quality

- All agent code must pass: `uv run ruff check .`
- All agent code must be formatted with: `uv run black .`
- All agent changes must include or update tests: `uv run pytest -q`

### Error Handling

```python
import structlog

logger = structlog.get_logger(__name__)

async def resilient_agent_operation():
    """Example of proper error handling in agents."""
    try:
        result = await risky_operation()
        logger.info("Operation completed successfully", result=result)
        return result
    except Exception as e:
        logger.error("Operation failed", error=str(e), operation="risky_operation")
        # Implement fallback or recovery logic
        return await fallback_operation()
```

### Logging and Monitoring

```python
# Use structured logging
logger.info("Agent action started", action="optimize_system", target="performance")

# Include relevant context
logger.error("Provider error", provider="google_gemini", error=str(e), retry_count=3)

# Log performance metrics
logger.info("Task completed", duration=elapsed_time, task_id=task_id, success=True)
```

## Testing and Validation

### Agent Testing

Create tests for agent functionality:

```python
# tests/test_jules_agent.py
import pytest
from ai_orchestration.agents.jules import JulesAgent

@pytest.mark.asyncio
async def test_agent_system_snapshot():
    """Test agent can take system snapshots."""
    agent = JulesAgent()
    snapshot = await agent.take_system_snapshot()
    
    assert "orchestrator_metrics" in snapshot
    assert "timestamp" in snapshot
    assert snapshot["timestamp"] is not None
```

### Integration Testing

Test agent integration with providers:

```python
@pytest.mark.asyncio
async def test_agent_gemini_integration():
    """Test agent integration with Gemini provider."""
    # This test should work without API keys (import-only)
    from ai_orchestration.providers.google_gemini import GoogleGeminiProvider
    
    # Test import and basic instantiation
    provider_class = GoogleGeminiProvider
    assert provider_class is not None
```

## Deployment and Production

### Environment Variables

Set required environment variables for agent operation:

```bash
# Copy and configure
cp .env.example .env

# Essential variables
GOOGLE_API_KEY=your_actual_api_key
GOOGLE_GENAI_MODEL=gemini-1.5-flash
AI_ORCHESTRATION_LOG_LEVEL=INFO
```

### Monitoring and Alerting

Agents should implement health checks and monitoring:

```python
async def agent_health_check():
    """Agent health check implementation."""
    return {
        "status": "healthy",
        "agent_type": "jules_async",
        "active_tasks": len(current_tasks),
        "last_action": last_action_timestamp,
        "performance_metrics": {
            "success_rate": calculate_success_rate(),
            "average_response_time": calculate_avg_response_time()
        }
    }
```

## TODO and Future Enhancements

- [ ] Implement streaming support for real-time agent responses
- [ ] Add multi-provider routing for agent decision-making
- [ ] Implement agent learning and adaptation mechanisms
- [ ] Add agent collaboration and coordination protocols
- [ ] Integrate with Vertex AI for enterprise-grade deployments
- [ ] Add function calling support for enhanced agent capabilities
- [ ] Implement agent authentication and authorization
- [ ] Add rate limiting and quota management for agents