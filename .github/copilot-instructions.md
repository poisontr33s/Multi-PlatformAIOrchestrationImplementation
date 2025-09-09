# GitHub Copilot Instructions for Multi-Platform AI Orchestration

## Project Overview
This is a sophisticated multi-platform AI orchestration system that coordinates between GitHub Copilot Pro+, Google AI Pro/Ultra, Microsoft AI Pro, Jules Asynchronous Coding Agent, Firebase Studio, and local Gemma 3/GPT-OSS models.

## Code Style Guidelines
- Use type hints for all Python functions
- Implement comprehensive error handling with structured logging
- Follow async/await patterns for all I/O operations
- Use dependency injection for all external service integrations
- Implement circuit breaker patterns for API calls
- Include comprehensive docstrings with examples

## Architecture Principles
- Modular design with clear separation of concerns
- Event-driven architecture with message queues
- Graceful degradation and fallback mechanisms
- Resource optimization and GPU memory management
- Security-first design with proper authentication
- Observability through metrics, logs, and traces

## Integration Patterns
- OAuth 2.1 + PKCE for all external authentications
- Rate limiting and retry logic for API calls
- Webhook handling with signature verification
- Container orchestration with health checks
- Configuration management through environment variables
- Secrets management through secure key stores

## Testing Requirements
- Unit tests for all core functionality
- Integration tests for all external service connections
- Performance tests for resource utilization
- Security tests for authentication and authorization
- End-to-end tests for complete workflows

## Documentation Standards
- API documentation with OpenAPI/Swagger
- Architecture decision records (ADRs)
- Deployment and configuration guides
- Troubleshooting and monitoring guides
- Performance optimization guides

## Development Workflow
1. All changes should be implemented with proper error handling
2. Include comprehensive logging for debugging
3. Add appropriate tests for new functionality
4. Update documentation when adding new features
5. Use circuit breakers for external service calls
6. Implement proper timeout handling
7. Add performance monitoring where applicable

## Security Guidelines
- Never commit secrets or credentials
- Use environment variables for all configuration
- Implement proper input validation
- Use secure authentication flows
- Log security events appropriately
- Follow principle of least privilege

## Performance Considerations
- Optimize for GPU memory usage
- Implement proper caching strategies
- Use async patterns for I/O operations
- Monitor resource utilization
- Implement backpressure handling
- Use connection pooling for external services

## File Organization
- `/src/orchestration/` - Core orchestration logic
- `/src/agents/` - Agent-specific implementations
- `/src/integrations/` - Platform integrations
- `/src/auth/` - Authentication and authorization
- `/src/utils/` - Utility functions and helpers
- `/src/monitoring/` - Performance monitoring
- `/tests/` - Test files organized by type
- `/configs/` - Configuration files
- `/.github/` - GitHub workflows and templates