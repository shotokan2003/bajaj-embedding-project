"""
Background tasks: Handles periodic tasks in separate threads to improve performance.
"""
import threading
import asyncio
import time
import logging
from typing import Callable, Dict, Any, List, Optional
from functools import wraps
from app.cache import clear_stale_cache

logger = logging.getLogger(__name__)

# Dictionary to store background tasks
_background_tasks = {}

class BackgroundTaskManager:
    """Manages background tasks with cron-like functionality."""
    
    def __init__(self):
        self.tasks = {}
        self.event = threading.Event()
        self.thread = None
    
    def start(self):
        """Start the background task manager."""
        if self.thread is None or not self.thread.is_alive():
            self.event.clear()
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()
            logger.info("Background task manager started")
    
    def stop(self):
        """Stop the background task manager."""
        if self.thread and self.thread.is_alive():
            self.event.set()
            self.thread.join(timeout=5)
            logger.info("Background task manager stopped")
    
    def _run(self):
        """Run the background task manager loop."""
        while not self.event.is_set():
            now = time.time()
            for task_id, task in list(self.tasks.items()):
                if now >= task["next_run"]:
                    # Execute the task in a new thread
                    t = threading.Thread(
                        target=self._execute_task,
                        args=(task_id, task),
                        daemon=True
                    )
                    t.start()
                    
                    # Update next run time
                    task["next_run"] = now + task["interval"]
            
            # Sleep for a short interval before checking again
            time.sleep(1)
    
    def _execute_task(self, task_id: str, task: Dict[str, Any]):
        """Execute a task with error handling."""
        try:
            logger.debug(f"Running background task: {task_id}")
            result = task["func"](*task["args"], **task["kwargs"])
            
            # Handle asyncio coroutines
            if asyncio.iscoroutine(result):
                # Create a new event loop for this thread
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    loop.run_until_complete(result)
                finally:
                    loop.close()
            
            logger.debug(f"Background task {task_id} completed")
        except Exception as e:
            logger.error(f"Error in background task {task_id}: {str(e)}")
    
    def add_task(
        self,
        task_id: str,
        func: Callable,
        interval: int,
        args: List = None,
        kwargs: Dict = None
    ):
        """Add a task to the background task manager."""
        self.tasks[task_id] = {
            "func": func,
            "interval": interval,
            "args": args or [],
            "kwargs": kwargs or {},
            "next_run": time.time()
        }
        logger.info(f"Added background task: {task_id}, interval: {interval}s")
    
    def remove_task(self, task_id: str):
        """Remove a task from the background task manager."""
        if task_id in self.tasks:
            del self.tasks[task_id]
            logger.info(f"Removed background task: {task_id}")

# Create a global task manager
task_manager = BackgroundTaskManager()

# Define default tasks
async def clean_cache_task():
    """Task to clean stale cache entries."""
    removed = clear_stale_cache(max_age_days=7)
    logger.info(f"Cleared {removed} stale cache entries")

def start_background_tasks():
    """Start background tasks for the application."""
    task_manager.start()
    
    # Add cache cleaning task - run every hour
    task_manager.add_task("cache_cleanup", clean_cache_task, 3600)
    
    logger.info("Background tasks initialized")

def stop_background_tasks():
    """Stop all background tasks."""
    task_manager.stop()
