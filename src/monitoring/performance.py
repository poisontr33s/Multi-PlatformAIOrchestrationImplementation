"""
Advanced performance monitoring and orchestration for the AI system.
"""

import asyncio
import time
import psutil
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import structlog

from ..orchestration.core import TaskExecution


@dataclass
class PerformanceMetrics:
    """Performance metrics for the system."""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    gpu_utilization: Optional[float]
    gpu_memory_used: Optional[float]
    active_tasks: int
    completed_tasks: int
    error_rate: float
    average_response_time: float


@dataclass
class ResourceUtilization:
    """Resource utilization metrics."""
    cpu_cores_used: int
    memory_mb_used: float
    gpu_memory_mb_used: Optional[float]
    disk_io_read: float
    disk_io_write: float
    network_bytes_sent: float
    network_bytes_recv: float


class AdvancedPerformanceOrchestrator:
    """
    Advanced performance monitoring with predictive optimization
    and automatic bottleneck prevention.
    """
    
    def __init__(self):
        self.logger = structlog.get_logger("performance_monitor")
        
        # Metrics storage
        self.metrics_history: List[PerformanceMetrics] = []
        self.resource_history: List[ResourceUtilization] = []
        
        # Performance tracking
        self.task_executions: List[TaskExecution] = []
        self.response_times: List[float] = []
        self.error_count = 0
        self.total_requests = 0
        
        # Monitoring state
        self._monitoring = False
        self._monitor_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Start the performance monitoring system."""
        try:
            self.logger.info("Starting advanced performance monitoring")
            
            self._monitoring = True
            self._monitor_task = asyncio.create_task(self._monitoring_loop())
            
            self.logger.info("Performance monitoring started successfully")
            
        except Exception as e:
            self.logger.error("Failed to start performance monitoring", error=str(e))
            raise

    async def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self._monitoring:
            try:
                # Collect system metrics
                await self._collect_system_metrics()
                
                # Collect resource utilization
                await self._collect_resource_utilization()
                
                # Analyze performance trends
                await self._analyze_performance_trends()
                
                # Check for bottlenecks
                await self._check_for_bottlenecks()
                
                # Wait before next collection
                await asyncio.sleep(30)  # Collect every 30 seconds
                
            except Exception as e:
                self.logger.error("Error in monitoring loop", error=str(e))
                await asyncio.sleep(60)  # Wait longer on error

    async def _collect_system_metrics(self) -> None:
        """Collect system performance metrics."""
        try:
            # Get CPU and memory usage
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            # Try to get GPU metrics (requires nvidia-ml-py)
            gpu_utilization = None
            gpu_memory_used = None
            
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                gpu_info = pynvml.nvmlDeviceGetUtilizationRates(handle)
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                
                gpu_utilization = gpu_info.gpu
                gpu_memory_used = (memory_info.used / memory_info.total) * 100
                
            except ImportError:
                self.logger.debug("pynvml not available, skipping GPU metrics")
            except Exception as e:
                self.logger.debug("Failed to get GPU metrics", error=str(e))
            
            # Calculate error rate
            error_rate = (self.error_count / max(self.total_requests, 1)) * 100
            
            # Calculate average response time
            avg_response_time = (
                sum(self.response_times[-100:]) / len(self.response_times[-100:])
                if self.response_times else 0.0
            )
            
            # Create metrics record
            metrics = PerformanceMetrics(
                timestamp=time.time(),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                gpu_utilization=gpu_utilization,
                gpu_memory_used=gpu_memory_used,
                active_tasks=0,  # Would be populated from orchestrator
                completed_tasks=len(self.task_executions),
                error_rate=error_rate,
                average_response_time=avg_response_time
            )
            
            # Store metrics
            self.metrics_history.append(metrics)
            
            # Keep only last 1000 entries
            if len(self.metrics_history) > 1000:
                self.metrics_history = self.metrics_history[-1000:]
                
        except Exception as e:
            self.logger.error("Failed to collect system metrics", error=str(e))

    async def _collect_resource_utilization(self) -> None:
        """Collect detailed resource utilization metrics."""
        try:
            # Get CPU info
            cpu_count = psutil.cpu_count()
            cpu_percent = psutil.cpu_percent()
            cpu_cores_used = int((cpu_percent / 100) * cpu_count)
            
            # Get memory info
            memory = psutil.virtual_memory()
            memory_mb_used = memory.used / (1024 * 1024)
            
            # Get disk I/O
            disk_io = psutil.disk_io_counters()
            disk_io_read = disk_io.read_bytes if disk_io else 0
            disk_io_write = disk_io.write_bytes if disk_io else 0
            
            # Get network I/O
            network_io = psutil.net_io_counters()
            network_bytes_sent = network_io.bytes_sent if network_io else 0
            network_bytes_recv = network_io.bytes_recv if network_io else 0
            
            # Try to get GPU memory
            gpu_memory_mb_used = None
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                gpu_memory_mb_used = memory_info.used / (1024 * 1024)
            except:
                pass
            
            # Create utilization record
            utilization = ResourceUtilization(
                cpu_cores_used=cpu_cores_used,
                memory_mb_used=memory_mb_used,
                gpu_memory_mb_used=gpu_memory_mb_used,
                disk_io_read=disk_io_read,
                disk_io_write=disk_io_write,
                network_bytes_sent=network_bytes_sent,
                network_bytes_recv=network_bytes_recv
            )
            
            # Store utilization
            self.resource_history.append(utilization)
            
            # Keep only last 1000 entries
            if len(self.resource_history) > 1000:
                self.resource_history = self.resource_history[-1000:]
                
        except Exception as e:
            self.logger.error("Failed to collect resource utilization", error=str(e))

    async def _analyze_performance_trends(self) -> None:
        """Analyze performance trends and patterns."""
        if len(self.metrics_history) < 10:
            return
        
        try:
            recent_metrics = self.metrics_history[-10:]
            
            # Calculate trends
            cpu_trend = self._calculate_trend([m.cpu_percent for m in recent_metrics])
            memory_trend = self._calculate_trend([m.memory_percent for m in recent_metrics])
            response_time_trend = self._calculate_trend([m.average_response_time for m in recent_metrics])
            
            # Log significant trends
            if cpu_trend > 5:
                self.logger.warning("CPU usage trending upward", trend=cpu_trend)
            if memory_trend > 5:
                self.logger.warning("Memory usage trending upward", trend=memory_trend)
            if response_time_trend > 10:
                self.logger.warning("Response time trending upward", trend=response_time_trend)
                
        except Exception as e:
            self.logger.error("Failed to analyze performance trends", error=str(e))

    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend direction (positive = increasing, negative = decreasing)."""
        if len(values) < 2:
            return 0.0
        
        # Simple linear trend calculation
        n = len(values)
        x_sum = sum(range(n))
        y_sum = sum(values)
        xy_sum = sum(i * values[i] for i in range(n))
        x2_sum = sum(i * i for i in range(n))
        
        try:
            slope = (n * xy_sum - x_sum * y_sum) / (n * x2_sum - x_sum * x_sum)
            return slope
        except ZeroDivisionError:
            return 0.0

    async def _check_for_bottlenecks(self) -> None:
        """Check for system bottlenecks and performance issues."""
        if not self.metrics_history:
            return
        
        try:
            latest_metrics = self.metrics_history[-1]
            
            # Check CPU bottleneck
            if latest_metrics.cpu_percent > 90:
                self.logger.warning("CPU bottleneck detected", cpu_percent=latest_metrics.cpu_percent)
                await self._handle_cpu_bottleneck()
            
            # Check memory bottleneck
            if latest_metrics.memory_percent > 90:
                self.logger.warning("Memory bottleneck detected", memory_percent=latest_metrics.memory_percent)
                await self._handle_memory_bottleneck()
            
            # Check GPU bottleneck
            if latest_metrics.gpu_utilization and latest_metrics.gpu_utilization > 95:
                self.logger.warning("GPU bottleneck detected", gpu_utilization=latest_metrics.gpu_utilization)
                await self._handle_gpu_bottleneck()
            
            # Check response time issues
            if latest_metrics.average_response_time > 10.0:
                self.logger.warning("High response time detected", response_time=latest_metrics.average_response_time)
                await self._handle_response_time_issue()
                
        except Exception as e:
            self.logger.error("Failed to check for bottlenecks", error=str(e))

    async def _handle_cpu_bottleneck(self) -> None:
        """Handle CPU bottleneck."""
        self.logger.info("Implementing CPU bottleneck mitigation")
        # Implementation would include task throttling, load balancing, etc.

    async def _handle_memory_bottleneck(self) -> None:
        """Handle memory bottleneck."""
        self.logger.info("Implementing memory bottleneck mitigation")
        # Implementation would include cache clearing, task prioritization, etc.

    async def _handle_gpu_bottleneck(self) -> None:
        """Handle GPU bottleneck."""
        self.logger.info("Implementing GPU bottleneck mitigation")
        # Implementation would include model switching, batch size adjustment, etc.

    async def _handle_response_time_issue(self) -> None:
        """Handle high response time issues."""
        self.logger.info("Implementing response time optimization")
        # Implementation would include timeout adjustment, circuit breaker tuning, etc.

    async def record_execution(self, execution: TaskExecution) -> None:
        """Record a task execution for performance analysis."""
        self.task_executions.append(execution)
        
        if execution.status == "completed":
            self.response_times.append(execution.execution_time)
        elif execution.status == "failed":
            self.error_count += 1
        
        self.total_requests += 1

    def get_current_metrics(self) -> Optional[PerformanceMetrics]:
        """Get the most recent performance metrics."""
        return self.metrics_history[-1] if self.metrics_history else None

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get a summary of performance metrics."""
        if not self.metrics_history:
            return {}
        
        recent_metrics = self.metrics_history[-10:] if len(self.metrics_history) >= 10 else self.metrics_history
        
        return {
            "current": {
                "cpu_percent": recent_metrics[-1].cpu_percent,
                "memory_percent": recent_metrics[-1].memory_percent,
                "gpu_utilization": recent_metrics[-1].gpu_utilization,
                "active_tasks": recent_metrics[-1].active_tasks,
                "error_rate": recent_metrics[-1].error_rate,
                "avg_response_time": recent_metrics[-1].average_response_time
            },
            "averages": {
                "cpu_percent": sum(m.cpu_percent for m in recent_metrics) / len(recent_metrics),
                "memory_percent": sum(m.memory_percent for m in recent_metrics) / len(recent_metrics),
                "avg_response_time": sum(m.average_response_time for m in recent_metrics) / len(recent_metrics)
            },
            "totals": {
                "completed_tasks": len(self.task_executions),
                "total_requests": self.total_requests,
                "error_count": self.error_count
            }
        }

    async def shutdown(self) -> None:
        """Shutdown the performance monitoring system."""
        self.logger.info("Shutting down performance monitoring")
        
        self._monitoring = False
        
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("Performance monitoring shutdown complete")