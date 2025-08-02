"""
Redis connection manager: Handles Redis connection with automatic reconnection.
"""

import os
import logging
import time
import redis
from functools import wraps

logger = logging.getLogger(__name__)

# Constants
REDIS_POOL_MAX_CONNECTIONS = 10
REDIS_CONNECT_TIMEOUT = 2.0  # seconds
REDIS_READ_TIMEOUT = 2.0  # seconds
MAX_RETRY_ATTEMPTS = 3
RETRY_DELAY = 0.2  # seconds

class RedisManager:
    """Redis connection manager with automatic reconnection"""
    
    def __init__(self):
        self.redis_url = os.environ.get("REDIS_URL")
        self.pool = None
        self.client = None
        self._initialize()
    
    def _initialize(self):
        """Initialize Redis connection pool"""
        if not self.redis_url:
            logger.warning("REDIS_URL not set")
            return
            
        try:
            self.pool = redis.ConnectionPool.from_url(
                self.redis_url,
                max_connections=REDIS_POOL_MAX_CONNECTIONS,
                socket_connect_timeout=REDIS_CONNECT_TIMEOUT,
                socket_timeout=REDIS_READ_TIMEOUT,
                retry_on_timeout=True
            )
            self.client = redis.Redis(connection_pool=self.pool)
            # Test connection
            self.client.ping()
            logger.info("Redis connection pool initialized")
        except (redis.RedisError, Exception) as e:
            logger.error(f"Redis initialization error: {str(e)}")
            self.pool = None
            self.client = None
    
    def get_client(self):
        """Get Redis client with connection check"""
        if not self.client:
            self._initialize()
            if not self.client:
                return None
                
        try:
            # Quick connection check
            self.client.ping()
            return self.client
        except (redis.ConnectionError, redis.TimeoutError) as e:
            logger.warning(f"Redis connection lost: {str(e)}. Reconnecting...")
            self._initialize()
            return self.client
        except Exception as e:
            logger.error(f"Redis error: {str(e)}")
            return None

# Create singleton instance
redis_manager = RedisManager()

def with_redis_retry(max_attempts=MAX_RETRY_ATTEMPTS, retry_delay=RETRY_DELAY):
    """Decorator to retry Redis operations with exponential backoff"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(1, max_attempts + 1):
                try:
                    client = redis_manager.get_client()
                    if not client:
                        if attempt == max_attempts:
                            raise Exception("Redis client unavailable")
                        time.sleep(retry_delay * attempt)
                        continue
                        
                    kwargs['client'] = client
                    return func(*args, **kwargs)
                except (redis.ConnectionError, redis.TimeoutError) as e:
                    last_error = e
                    if attempt < max_attempts:
                        logger.debug(f"Redis operation failed (attempt {attempt}/{max_attempts}): {str(e)}")
                        time.sleep(retry_delay * attempt)
                    else:
                        logger.error(f"Redis operation failed after {max_attempts} attempts: {str(e)}")
                except Exception as e:
                    logger.error(f"Redis operation error: {str(e)}")
                    last_error = e
                    break
            
            # Fall back to None for get operations
            return None
        return wrapper
    return decorator
