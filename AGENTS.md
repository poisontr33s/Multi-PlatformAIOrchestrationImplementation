# AGENTS.md - AI Agent Integration Guide

## Overview

This document provides guidance for AI agents (including Jules and other coding assistants) on how to effectively work with the Multi-Platform AI Orchestration system.

## Quick Start for AI Agents

### Environment Setup

1. **Check if you're in a Jules environment:**
   ```bash
   # Jules environments have these indicators
   [ -n "$JULES_ENV" ] || [ -n "$CODESPACE_NAME" ] || [ -f "/.dockerenv" ]
   ```

2. **Run the appropriate setup script:**
   ```bash
   # For Jules VMs (recommended)
   ./setup-jules.sh
   
   # For local development
   ./setup-local.sh
   ```

3. **Activate the environment:**
   ```bash
   source activate.sh
   ```

### Available Tools

#### Python CLI (Primary Interface)
```bash
# Initialize the orchestration system
ai-orchestrator init --subscription --emergent --verbose

# Check system status
ai-orchestrator status

# Submit tasks
ai-orchestrator submit-task my_task_001 \
  --type autonomous_strategic_implementation \
  --priority critical_infrastructure \
  --payload '{"objective": "optimize_performance"}'

# View discovered patterns
ai-orchestrator patterns
```

#### Node.js CLI Bridges
```bash
# Google Gemini
bun run gemini --prompt "Explain AI orchestration" --model gemini-1.5-pro

# Anthropic Claude
bun run claude --prompt "Review this architecture" --model claude-3-5-sonnet-20241022

# OpenAI GPT
bun run openai --prompt "Generate code" --model gpt-4-turbo-preview
```

#### REST API
```bash
# Start the API server
bun run dev
# or
python -m uvicorn ai_orchestration.api:app --reload

# Use the API
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4",
    "messages": [
      {"role": "user", "content": "Hello!"}
    ]
  }'
```

## AI Provider Configuration

### Required Environment Variables

Set these environment variables for the providers you want to use:

```bash
# Google Gemini
export GOOGLE_API_KEY="your_google_api_key"
# or
export GEMINI_API_KEY="your_gemini_api_key"

# Anthropic Claude  
export ANTHROPIC_API_KEY="your_anthropic_api_key"
# or
export CLAUDE_API_KEY="your_claude_api_key"

# OpenAI GPT
export OPENAI_API_KEY="your_openai_api_key"

# Hugging Face (optional, for OSS models)
export HUGGINGFACE_API_KEY="your_hf_token"
# or
export HF_TOKEN="your_hf_token"
```

### Provider Selection

The system automatically detects available providers based on API keys. You can:

1. **Use default provider:** Most commands will use the first available provider
2. **Specify provider explicitly:** Add `--provider openai` to CLI commands
3. **Use REST API with provider:** Include `"provider": "anthropic"` in API requests

## Common Patterns for AI Agents

### 1. Code Analysis and Review

```bash
# Use Claude for code analysis (excellent reasoning)
bun run claude --file code_to_review.py --system "You are a senior code reviewer. Analyze this code for bugs, performance issues, and best practices."

# Use GPT-4 for architectural review
bun run openai --prompt "Review this system architecture" --model gpt-4 --file architecture.md
```

### 2. Content Generation

```bash
# Use Gemini for creative content
bun run gemini --prompt "Write documentation for this API" --model gemini-1.5-pro

# Use Claude for technical writing
bun run claude --prompt "Explain this complex algorithm" --model claude-3-5-sonnet-20241022
```

### 3. Problem Solving with Orchestration

```python
# Python API for complex orchestration
import asyncio
from ai_orchestration import (
    Orchestrator, OrchestrationTask, TaskClassification, 
    PriorityLevel, ComplexityRating
)

async def solve_complex_problem():
    orchestrator = Orchestrator()
    await orchestrator.initialize()
    
    task = OrchestrationTask(
        id="problem_solving_001",
        classification=TaskClassification.AUTONOMOUS_STRATEGIC_IMPLEMENTATION,
        priority=PriorityLevel.CRITICAL_INFRASTRUCTURE,
        complexity=ComplexityRating.ARCHITECTURAL_SYNTHESIS,
        payload={
            "problem": "Optimize system performance",
            "constraints": {"budget": 1000, "time": "1 week"},
            "objectives": ["reduce latency", "increase throughput"]
        }
    )
    
    result = await orchestrator.submit_task(task)
    return result
```

### 4. Multi-Provider Comparison

```bash
# Compare responses from different providers
echo "Explain quantum computing" > prompt.txt

echo "=== Google Gemini ==="
bun run gemini --file prompt.txt --model gemini-1.5-pro

echo "=== Anthropic Claude ==="  
bun run claude --file prompt.txt --model claude-3-5-sonnet-20241022

echo "=== OpenAI GPT ==="
bun run openai --file prompt.txt --model gpt-4-turbo-preview
```

## Best Practices for AI Agents

### 1. Provider Selection Strategy

- **Claude 3.5 Sonnet**: Best for code analysis, reasoning, complex problem-solving
- **GPT-4**: Excellent general-purpose model, good for creative tasks
- **Gemini 1.5 Pro**: Great for long context, technical documentation
- **Hugging Face OSS**: Use for privacy-sensitive tasks or offline scenarios

### 2. Error Handling

```bash
# Check system health before operations
ai-orchestrator status

# Verify API connectivity
curl -s http://localhost:8000/health | jq '.status'

# Graceful fallbacks
bun run openai --prompt "test" || bun run claude --prompt "test" || echo "All providers failed"
```

### 3. Performance Optimization

```bash
# Use streaming for long responses
bun run claude --prompt "Write a detailed analysis" --stream

# Batch similar requests
ai-orchestrator submit-task batch_001 --payload '{"requests": [...]}' 
```

### 4. Monitoring and Logging

```bash
# Enable verbose logging
export AI_ORCHESTRATION_LOG_LEVEL=DEBUG

# Monitor system metrics
ai-orchestrator status | jq '.metrics'

# View discovered patterns
ai-orchestrator patterns
```

## Integration Examples

### Jules-Specific Integration

```bash
#!/bin/bash
# Jules auto-setup script

# Detect Jules environment
if [[ -n "$JULES_ENV" || -n "$CODESPACE_NAME" ]]; then
    echo "🚀 Jules environment detected, setting up AI orchestration..."
    
    # Run Jules-optimized setup
    ./setup-jules.sh
    
    # Activate environment
    source activate.sh
    
    # Quick health check
    ai-orchestrator status
    
    echo "✅ Ready for AI-assisted development!"
fi
```

### API Integration for Web Interfaces

```javascript
// Example: Integrate with web UI
const response = await fetch('http://localhost:8000/chat', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    model: 'claude-3-5-sonnet-20241022',
    messages: [
      { role: 'user', content: 'Help me debug this code' }
    ],
    max_tokens: 1000,
    temperature: 0.1
  })
});

const result = await response.json();
console.log(result.message.content);
```

### Workflow Automation

```bash
#!/bin/bash
# Automated code review workflow

# 1. Analyze code with Claude
ANALYSIS=$(bun run claude --file "$1" --system "Analyze this code for issues" --max-tokens 500)

# 2. Generate suggestions with GPT-4
SUGGESTIONS=$(bun run openai --prompt "Based on this analysis: $ANALYSIS, provide specific improvement suggestions" --model gpt-4)

# 3. Submit to orchestration system
ai-orchestrator submit-task "code_review_$(date +%s)" \
  --payload "{\"file\":\"$1\", \"analysis\":\"$ANALYSIS\", \"suggestions\":\"$SUGGESTIONS\"}"
```

## Troubleshooting

### Common Issues

1. **No providers available**
   ```bash
   # Check API keys
   env | grep -E "(GOOGLE|ANTHROPIC|OPENAI|HUGGINGFACE).*API"
   
   # Test connectivity
   bun run setup-checks
   ```

2. **Import errors**
   ```bash
   # Reinstall dependencies
   pip install -e .
   bun install
   ```

3. **CLI not found**
   ```bash
   # Activate environment
   source activate.sh
   
   # Check PATH
   which ai-orchestrator
   ```

### Getting Help

- **Check system status:** `ai-orchestrator status`
- **View logs:** Check `logs/` directory or set `AI_ORCHESTRATION_LOG_LEVEL=DEBUG`
- **API documentation:** Visit `http://localhost:8000/docs` when server is running
- **Health check:** `curl http://localhost:8000/health`

## Advanced Usage

### Custom Provider Integration

See `src/ai_orchestration/providers/` for examples of implementing custom providers.

### Extending the Orchestration System

The system is designed for extensibility. Key extension points:

- **Custom agents:** Extend `AgentCoordinator`
- **New task types:** Add to `TaskClassification`
- **Custom intelligence:** Extend `EmergentIntelligence`
- **API endpoints:** Add to `ai_orchestration.api`

### Performance Tuning

```bash
# Optimize for your workload
export AI_ORCHESTRATION_MAX_TASKS=50
export AI_ORCHESTRATION_MAX_AGENTS=20
export AI_ORCHESTRATION_COORDINATION_STRATEGY=emergent_adaptive
```

## Conclusion

The Multi-Platform AI Orchestration system provides a unified interface to multiple AI providers while maintaining the sophisticated orchestration capabilities of the original system. AI agents can leverage this to provide better assistance with automatic provider selection, intelligent load balancing, and emergent intelligence features.

For more detailed information, see `docs/INTEGRATIONS.md` and the API documentation at `/docs` when the server is running.