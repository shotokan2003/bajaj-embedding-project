"""
Cerebras batch manager: Handles rate limiting and batch processing for LLM requests
"""

import os
import time
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Optional, Callable, Tuple
import functools

logger = logging.getLogger(__name__)

# Rate limiting constants - ultra-fast optimization
MAX_QPS = 12  # Aggressive increase for speed  
REQUEST_COOLDOWN = 1.0 / MAX_QPS  # 83ms between requests
MAX_BATCH_SIZE = 8  # Larger batches for efficiency
MAX_WORKERS = 8  # More workers for parallel processing

class CerebrasBatchManager:
    """Manages rate limiting and batch processing for Cerebras API"""
    
    def __init__(self):
        self.last_request_time = 0
        self.lock = asyncio.Lock()  # To synchronize access to last_request_time
        
    async def process_batch(self, 
                      prompts: List[str], 
                      completion_func: Callable[[str], str]) -> List[str]:
        """Process a batch of prompts with rate limiting"""
        if not prompts:
            return []
        
        # Split into smaller batches
        batches = [prompts[i:i + MAX_BATCH_SIZE] 
                  for i in range(0, len(prompts), MAX_BATCH_SIZE)]
        
        # Process batches sequentially to control overall rate
        results = []
        for batch in batches:
            # Process items in the batch with limited concurrency
            batch_results = await self._process_with_rate_limit(batch, completion_func)
            results.extend(batch_results)
        
        return results
    
    async def _process_with_rate_limit(self, 
                               prompts: List[str], 
                               completion_func: Callable[[str], str]) -> List[str]:
        """Process a single batch with rate limiting"""
        loop = asyncio.get_event_loop()
        executor = ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(prompts)))
        tasks = []
        
        for prompt in prompts:
            # Ensure we don't exceed QPS by waiting if needed
            await self._wait_for_rate_limit()
            
            # Run the completion in a thread pool
            task = loop.run_in_executor(
                executor,
                completion_func,
                prompt
            )
            tasks.append(task)
        
        # Wait for all completions in this batch
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions
        processed_results = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Error in batch processing: {str(result)}")
                processed_results.append("")
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def _wait_for_rate_limit(self):
        """Wait if needed to respect rate limits"""
        async with self.lock:
            now = time.time()
            time_since_last = now - self.last_request_time
            
            if time_since_last < REQUEST_COOLDOWN:
                wait_time = REQUEST_COOLDOWN - time_since_last
                await asyncio.sleep(wait_time)
            
            self.last_request_time = time.time()

# Global instance
batch_manager = CerebrasBatchManager()
