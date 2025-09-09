"""
Retry manager utility for handling transient failures.
Implements exponential backoff with jitter for robust retry logic.
"""

import asyncio
import random
import time
from typing import Callable, Any, Optional, List, Type
from dataclasses import dataclass
from enum import Enum
import structlog


class RetryStrategy(Enum):
    """Retry strategies."""
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    FIXED_DELAY = "fixed_delay"
    IMMEDIATE = "immediate"


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    backoff_multiplier: float = 2.0
    jitter: bool = True
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    retryable_exceptions: List[Type[Exception]] = None


@dataclass
class RetryAttempt:
    """Information about a retry attempt."""
    attempt_number: int
    delay: float
    exception: Optional[Exception]
    timestamp: float


class RetryExhaustedException(Exception):
    """Exception raised when all retry attempts are exhausted."""
    
    def __init__(self, attempts: List[RetryAttempt], last_exception: Exception):
        self.attempts = attempts
        self.last_exception = last_exception
        super().__init__(f"Retry exhausted after {len(attempts)} attempts. Last error: {last_exception}")


class RetryManager:
    """
    Retry manager with configurable strategies and exponential backoff.
    
    Provides robust retry logic with circuit breaker integration and
    detailed metrics collection.
    """
    
    def __init__(self, 
                 max_attempts: int = 3,
                 base_delay: float = 1.0,
                 max_delay: float = 60.0,
                 backoff_multiplier: float = 2.0,
                 jitter: bool = True,
                 strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF,
                 retryable_exceptions: Optional[List[Type[Exception]]] = None,
                 name: Optional[str] = None):
        """
        Initialize retry manager.
        
        Args:
            max_attempts: Maximum number of retry attempts
            base_delay: Base delay between retries (seconds)
            max_delay: Maximum delay between retries (seconds)
            backoff_multiplier: Multiplier for exponential backoff
            jitter: Whether to add random jitter to delays
            strategy: Retry strategy to use
            retryable_exceptions: List of exceptions that should trigger retries
            name: Optional name for logging
        """
        self.config = RetryConfig(
            max_attempts=max_attempts,
            base_delay=base_delay,
            max_delay=max_delay,
            backoff_multiplier=backoff_multiplier,
            jitter=jitter,
            strategy=strategy,
            retryable_exceptions=retryable_exceptions or [Exception]
        )
        self.name = name or "retry_manager"
        self.logger = structlog.get_logger("retry_manager", name=self.name)

    async def execute(self, func: Callable[[], Any]) -> Any:
        """
        Execute a function with retry logic.
        
        Args:
            func: Function to execute (can be sync or async)
            
        Returns:
            Function result
            
        Raises:
            RetryExhaustedException: When all retry attempts are exhausted
        """
        attempts: List[RetryAttempt] = []
        last_exception = None
        
        for attempt_num in range(1, self.config.max_attempts + 1):
            try:
                # Log attempt
                if attempt_num > 1:
                    self.logger.info("Retrying function execution", 
                                   attempt=attempt_num,
                                   max_attempts=self.config.max_attempts,
                                   name=self.name)
                
                # Execute the function
                if asyncio.iscoroutinefunction(func):
                    result = await func()
                else:
                    result = func()
                
                # Success - log and return
                if attempt_num > 1:
                    self.logger.info("Function succeeded after retry", 
                                   attempt=attempt_num,
                                   name=self.name)
                
                return result
                
            except Exception as e:
                last_exception = e
                
                # Check if this exception should trigger a retry
                if not self._should_retry(e):
                    self.logger.info("Exception not retryable, failing immediately", 
                                   exception_type=type(e).__name__,
                                   name=self.name)
                    raise e
                
                # Record attempt
                attempt = RetryAttempt(
                    attempt_number=attempt_num,
                    delay=0.0,
                    exception=e,
                    timestamp=time.time()
                )
                attempts.append(attempt)
                
                # Check if this was the last attempt
                if attempt_num >= self.config.max_attempts:
                    self.logger.error("All retry attempts exhausted", 
                                    attempts=len(attempts),
                                    last_exception=str(e),
                                    name=self.name)
                    raise RetryExhaustedException(attempts, e)
                
                # Calculate delay for next attempt
                delay = self._calculate_delay(attempt_num)
                attempt.delay = delay
                
                self.logger.warning("Function failed, will retry", 
                                  attempt=attempt_num,
                                  max_attempts=self.config.max_attempts,
                                  delay=delay,
                                  exception=str(e),
                                  name=self.name)
                
                # Wait before retry
                await asyncio.sleep(delay)
        
        # Should never reach here, but just in case
        raise RetryExhaustedException(attempts, last_exception)

    def _should_retry(self, exception: Exception) -> bool:
        """Check if an exception should trigger a retry."""
        if not self.config.retryable_exceptions:
            return True
        
        return any(isinstance(exception, exc_type) 
                  for exc_type in self.config.retryable_exceptions)

    def _calculate_delay(self, attempt_number: int) -> float:
        """Calculate delay for the given attempt number."""
        if self.config.strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            delay = self.config.base_delay * (self.config.backoff_multiplier ** (attempt_number - 1))
        elif self.config.strategy == RetryStrategy.LINEAR_BACKOFF:
            delay = self.config.base_delay * attempt_number
        elif self.config.strategy == RetryStrategy.FIXED_DELAY:
            delay = self.config.base_delay
        elif self.config.strategy == RetryStrategy.IMMEDIATE:
            delay = 0.0
        else:
            delay = self.config.base_delay
        
        # Apply maximum delay limit
        delay = min(delay, self.config.max_delay)
        
        # Add jitter if enabled
        if self.config.jitter and delay > 0:
            jitter_amount = delay * 0.1  # 10% jitter
            delay += random.uniform(-jitter_amount, jitter_amount)
            delay = max(0, delay)  # Ensure non-negative
        
        return delay

    async def execute_with_timeout(self, func: Callable[[], Any], timeout: float) -> Any:
        """
        Execute a function with retry logic and timeout.
        
        Args:
            func: Function to execute
            timeout: Timeout for the entire retry operation (seconds)
            
        Returns:
            Function result
            
        Raises:
            asyncio.TimeoutError: When timeout is exceeded
            RetryExhaustedException: When all retry attempts are exhausted
        """
        return await asyncio.wait_for(self.execute(func), timeout=timeout)

    def with_config(self, **kwargs) -> 'RetryManager':
        """Create a new RetryManager with modified configuration."""
        new_config = RetryConfig(
            max_attempts=kwargs.get('max_attempts', self.config.max_attempts),
            base_delay=kwargs.get('base_delay', self.config.base_delay),
            max_delay=kwargs.get('max_delay', self.config.max_delay),
            backoff_multiplier=kwargs.get('backoff_multiplier', self.config.backoff_multiplier),
            jitter=kwargs.get('jitter', self.config.jitter),
            strategy=kwargs.get('strategy', self.config.strategy),
            retryable_exceptions=kwargs.get('retryable_exceptions', self.config.retryable_exceptions)
        )
        
        return RetryManager(
            max_attempts=new_config.max_attempts,
            base_delay=new_config.base_delay,
            max_delay=new_config.max_delay,
            backoff_multiplier=new_config.backoff_multiplier,
            jitter=new_config.jitter,
            strategy=new_config.strategy,
            retryable_exceptions=new_config.retryable_exceptions,
            name=self.name
        )

    @staticmethod
    def exponential_backoff(max_attempts: int = 3, 
                          base_delay: float = 1.0,
                          max_delay: float = 60.0) -> 'RetryManager':
        """Create a RetryManager with exponential backoff strategy."""
        return RetryManager(
            max_attempts=max_attempts,
            base_delay=base_delay,
            max_delay=max_delay,
            strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
            name="exponential_backoff"
        )

    @staticmethod
    def linear_backoff(max_attempts: int = 3, 
                      base_delay: float = 1.0,
                      max_delay: float = 30.0) -> 'RetryManager':
        """Create a RetryManager with linear backoff strategy."""
        return RetryManager(
            max_attempts=max_attempts,
            base_delay=base_delay,
            max_delay=max_delay,
            strategy=RetryStrategy.LINEAR_BACKOFF,
            name="linear_backoff"
        )

    @staticmethod
    def fixed_delay(max_attempts: int = 3, 
                   delay: float = 1.0) -> 'RetryManager':
        """Create a RetryManager with fixed delay strategy."""
        return RetryManager(
            max_attempts=max_attempts,
            base_delay=delay,
            max_delay=delay,
            strategy=RetryStrategy.FIXED_DELAY,
            name="fixed_delay"
        )

    @staticmethod
    def immediate(max_attempts: int = 3) -> 'RetryManager':
        """Create a RetryManager with immediate retry strategy."""
        return RetryManager(
            max_attempts=max_attempts,
            base_delay=0.0,
            max_delay=0.0,
            strategy=RetryStrategy.IMMEDIATE,
            name="immediate"
        )

    def get_stats(self) -> dict:
        """Get retry manager statistics."""
        return {
            "name": self.name,
            "config": {
                "max_attempts": self.config.max_attempts,
                "base_delay": self.config.base_delay,
                "max_delay": self.config.max_delay,
                "backoff_multiplier": self.config.backoff_multiplier,
                "jitter": self.config.jitter,
                "strategy": self.config.strategy.value,
                "retryable_exceptions": [exc.__name__ for exc in (self.config.retryable_exceptions or [])]
            }
        }