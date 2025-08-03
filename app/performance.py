"""
Performance monitoring module for tracking chunking and embedding optimization.
Provides metrics and insights for continuous improvement.
"""

import time
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import statistics
import asyncio
from functools import wraps

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Data class to store performance metrics"""
    operation: str
    start_time: float
    end_time: float
    duration: float
    success: bool
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration_ms(self) -> float:
        """Duration in milliseconds"""
        return self.duration * 1000

class PerformanceMonitor:
    """Monitor and track performance of chunking and embedding operations"""
    
    def __init__(self):
        self.metrics: List[PerformanceMetrics] = []
        self.operation_stats: Dict[str, Dict[str, float]] = {}
    
    def record_operation(self, operation: str, duration: float, success: bool = True, **metadata):
        """Record a performance metric"""
        metric = PerformanceMetrics(
            operation=operation,
            start_time=time.time() - duration,
            end_time=time.time(),
            duration=duration,
            success=success,
            metadata=metadata
        )
        self.metrics.append(metric)
        self._update_stats(operation, duration, success)
        
        # Log slow operations
        if duration > 5.0:  # More than 5 seconds
            logger.warning(f"Slow operation detected: {operation} took {duration:.2f}s")
    
    def _update_stats(self, operation: str, duration: float, success: bool):
        """Update operation statistics"""
        if operation not in self.operation_stats:
            self.operation_stats[operation] = {
                'count': 0,
                'total_time': 0,
                'avg_time': 0,
                'min_time': float('inf'),
                'max_time': 0,
                'success_rate': 0,
                'success_count': 0
            }
        
        stats = self.operation_stats[operation]
        stats['count'] += 1
        stats['total_time'] += duration
        stats['avg_time'] = stats['total_time'] / stats['count']
        stats['min_time'] = min(stats['min_time'], duration)
        stats['max_time'] = max(stats['max_time'], duration)
        
        if success:
            stats['success_count'] += 1
        stats['success_rate'] = stats['success_count'] / stats['count']
    
    def get_operation_stats(self, operation: str) -> Optional[Dict[str, float]]:
        """Get statistics for a specific operation"""
        return self.operation_stats.get(operation)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        if not self.metrics:
            return {"message": "No metrics recorded"}
        
        total_operations = len(self.metrics)
        successful_operations = sum(1 for m in self.metrics if m.success)
        total_time = sum(m.duration for m in self.metrics)
        
        durations = [m.duration for m in self.metrics if m.success]
        
        summary = {
            'total_operations': total_operations,
            'successful_operations': successful_operations,
            'success_rate': successful_operations / total_operations if total_operations > 0 else 0,
            'total_time': total_time,
            'average_duration': statistics.mean(durations) if durations else 0,
            'median_duration': statistics.median(durations) if durations else 0,
            'min_duration': min(durations) if durations else 0,
            'max_duration': max(durations) if durations else 0,
            'operation_breakdown': dict(self.operation_stats)
        }
        
        return summary
    
    def get_recommendations(self) -> List[str]:
        """Get performance optimization recommendations"""
        recommendations = []
        
        if not self.operation_stats:
            return ["No operations recorded for analysis"]
        
        # Check for slow operations
        for operation, stats in self.operation_stats.items():
            if stats['avg_time'] > 10.0:
                recommendations.append(
                    f"Operation '{operation}' is slow (avg: {stats['avg_time']:.2f}s). "
                    "Consider optimization."
                )
            
            if stats['success_rate'] < 0.95:
                recommendations.append(
                    f"Operation '{operation}' has low success rate ({stats['success_rate']:.2%}). "
                    "Check for errors."
                )
        
        # Check embedding performance
        if 'embedding_generation' in self.operation_stats:
            embed_stats = self.operation_stats['embedding_generation']
            if embed_stats['avg_time'] > 5.0:
                recommendations.append(
                    "Embedding generation is slow. Consider reducing batch size or "
                    "optimizing chunk sizes."
                )
        
        # Check chunking performance
        if 'text_chunking' in self.operation_stats:
            chunk_stats = self.operation_stats['text_chunking']
            if chunk_stats['avg_time'] > 2.0:
                recommendations.append(
                    "Text chunking is slow. Consider optimizing chunk size or "
                    "reducing overlap."
                )
        
        if not recommendations:
            recommendations.append("Performance looks good! No immediate optimizations needed.")
        
        return recommendations
    
    def reset_metrics(self):
        """Reset all recorded metrics"""
        self.metrics = []
        self.operation_stats = {}
        logger.info("Performance metrics reset")

# Global performance monitor instance
performance_monitor = PerformanceMonitor()

def monitor_performance(operation_name: str):
    """Decorator to monitor function performance"""
    def decorator(func):
        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = await func(*args, **kwargs)
                    duration = time.time() - start_time
                    performance_monitor.record_operation(
                        operation_name, 
                        duration, 
                        success=True,
                        args_count=len(args),
                        kwargs_keys=list(kwargs.keys())
                    )
                    return result
                except Exception as e:
                    duration = time.time() - start_time
                    performance_monitor.record_operation(
                        operation_name, 
                        duration, 
                        success=False,
                        error=str(e)
                    )
                    raise
            return async_wrapper
        else:
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    duration = time.time() - start_time
                    performance_monitor.record_operation(
                        operation_name, 
                        duration, 
                        success=True,
                        args_count=len(args),
                        kwargs_keys=list(kwargs.keys())
                    )
                    return result
                except Exception as e:
                    duration = time.time() - start_time
                    performance_monitor.record_operation(
                        operation_name, 
                        duration, 
                        success=False,
                        error=str(e)
                    )
                    raise
            return sync_wrapper
    return decorator

def log_performance_summary():
    """Log a summary of current performance metrics"""
    summary = performance_monitor.get_summary()
    recommendations = performance_monitor.get_recommendations()
    
    logger.info("=== Performance Summary ===")
    logger.info(f"Total operations: {summary.get('total_operations', 0)}")
    logger.info(f"Success rate: {summary.get('success_rate', 0):.2%}")
    logger.info(f"Average duration: {summary.get('average_duration', 0):.2f}s")
    
    logger.info("=== Recommendations ===")
    for rec in recommendations:
        logger.info(f"- {rec}")
