"""
Circuit breaker utility for handling service failures gracefully.
Implements the circuit breaker pattern for external service calls.
"""

import asyncio
import time
from typing import Callable, Any, Optional
from dataclasses import dataclass
from enum import Enum
import structlog


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"         # Circuit is open, calls fail immediately
    HALF_OPEN = "half_open"  # Testing if service is back


@dataclass
class CircuitMetrics:
    """Circuit breaker metrics."""
    total_calls: int = 0
    failed_calls: int = 0
    successful_calls: int = 0
    last_failure_time: Optional[float] = None
    last_success_time: Optional[float] = None


class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open."""
    pass


class CircuitBreaker:
    """
    Circuit breaker implementation for handling external service failures.
    
    Provides automatic failure detection and recovery with configurable
    thresholds and timeouts.
    """
    
    def __init__(self, 
                 failure_threshold: int = 5,
                 recovery_timeout: float = 60.0,
                 expected_exception: type = Exception,
                 name: Optional[str] = None):
        """
        Initialize circuit breaker.
        
        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Time to wait before trying again (seconds)
            expected_exception: Exception type that triggers circuit opening
            name: Optional name for logging
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.name = name or "circuit_breaker"
        
        self.state = CircuitState.CLOSED
        self.metrics = CircuitMetrics()
        self.logger = structlog.get_logger("circuit_breaker", name=self.name)
        
        # Thread safety
        self._lock = asyncio.Lock()

    async def call(self, func: Callable[[], Any]) -> Any:
        """
        Execute a function with circuit breaker protection.
        
        Args:
            func: Function to execute
            
        Returns:
            Function result
            
        Raises:
            CircuitBreakerOpenError: When circuit is open
            Exception: Original exception from function
        """
        async with self._lock:
            # Check circuit state
            await self._check_state()
            
            if self.state == CircuitState.OPEN:
                raise CircuitBreakerOpenError(
                    f"Circuit breaker '{self.name}' is open. "
                    f"Last failure: {self.metrics.last_failure_time}"
                )
            
            try:
                # Execute the function
                if asyncio.iscoroutinefunction(func):
                    result = await func()
                else:
                    result = func()
                
                # Record success
                await self._record_success()
                return result
                
            except self.expected_exception as e:
                # Record failure
                await self._record_failure()
                raise e

    async def _check_state(self) -> None:
        """Check and update circuit breaker state."""
        current_time = time.time()
        
        if self.state == CircuitState.OPEN:
            # Check if recovery timeout has passed
            if (self.metrics.last_failure_time and 
                current_time - self.metrics.last_failure_time >= self.recovery_timeout):
                self.state = CircuitState.HALF_OPEN
                self.logger.info("Circuit breaker moving to half-open state", 
                               name=self.name,
                               recovery_timeout=self.recovery_timeout)

    async def _record_success(self) -> None:
        """Record a successful call."""
        self.metrics.total_calls += 1
        self.metrics.successful_calls += 1
        self.metrics.last_success_time = time.time()
        
        # Reset circuit to closed if it was half-open
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.CLOSED
            self.metrics.failed_calls = 0  # Reset failure count
            self.logger.info("Circuit breaker closed after successful recovery", 
                           name=self.name)

    async def _record_failure(self) -> None:
        """Record a failed call."""
        self.metrics.total_calls += 1
        self.metrics.failed_calls += 1
        self.metrics.last_failure_time = time.time()
        
        # Check if we should open the circuit
        if (self.metrics.failed_calls >= self.failure_threshold and 
            self.state != CircuitState.OPEN):
            
            self.state = CircuitState.OPEN
            self.logger.warning("Circuit breaker opened due to failures", 
                              name=self.name,
                              failed_calls=self.metrics.failed_calls,
                              threshold=self.failure_threshold)

    async def reset(self) -> None:
        """Reset circuit breaker to closed state."""
        async with self._lock:
            self.state = CircuitState.CLOSED
            self.metrics = CircuitMetrics()
            self.logger.info("Circuit breaker manually reset", name=self.name)

    @property
    def is_open(self) -> bool:
        """Check if circuit breaker is open."""
        return self.state == CircuitState.OPEN

    @property
    def is_closed(self) -> bool:
        """Check if circuit breaker is closed."""
        return self.state == CircuitState.CLOSED

    @property
    def is_half_open(self) -> bool:
        """Check if circuit breaker is half-open."""
        return self.state == CircuitState.HALF_OPEN

    @property
    def failure_rate(self) -> float:
        """Calculate current failure rate."""
        if self.metrics.total_calls == 0:
            return 0.0
        return self.metrics.failed_calls / self.metrics.total_calls

    def get_metrics(self) -> dict:
        """Get circuit breaker metrics."""
        return {
            "name": self.name,
            "state": self.state.value,
            "total_calls": self.metrics.total_calls,
            "failed_calls": self.metrics.failed_calls,
            "successful_calls": self.metrics.successful_calls,
            "failure_rate": self.failure_rate,
            "last_failure_time": self.metrics.last_failure_time,
            "last_success_time": self.metrics.last_success_time,
            "failure_threshold": self.failure_threshold,
            "recovery_timeout": self.recovery_timeout
        }