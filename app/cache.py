"""
Cache module: Handles persistent disk caching, Redis and in-memory LRU for embeddings and answers.
Optimized for high-performance retrieval with minimal latency.
Supports Vercel deployment with Redis or in-memory cache only.
"""

import os
import pickle
import hashlib
import time
import json
import logging
from functools import lru_cache
from typing import Dict, Any, Optional, Tuple, Union
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Fix for NumPy 2.0+: ensure we use float64 instead of float_
if hasattr(np, 'float_'):
    np.float64 = np.float_

# Detect environment
IS_VERCEL = os.environ.get("VERCEL", "0") == "1"
USE_REDIS = os.environ.get("USE_REDIS", "0") == "1"

# Initialize Redis with connection manager if configured
if USE_REDIS:
    try:
        import redis
        from app.redis_manager import redis_manager, with_redis_retry
        logger.info("Redis cache initialized with connection manager")
    except ImportError:
        logger.warning("Redis package not installed but USE_REDIS=1. Falling back to local cache.")
        USE_REDIS = False

# In-memory cache for Vercel or fallback
MEMORY_CACHE = {
    "embeddings": {},
    "answers": {},
    "documents": {}
}

# Cache settings
CACHE_DIR = os.path.join(os.path.dirname(__file__), "../cache")
EMBEDDING_CACHE_DIR = os.path.join(CACHE_DIR, "embeddings")
ANSWER_CACHE_DIR = os.path.join(CACHE_DIR, "answers")
DOCUMENT_CACHE_DIR = os.path.join(CACHE_DIR, "documents")
LRU_MAXSIZE = 100  # Number of items to keep in LRU cache

# Create cache directories if not on Vercel and not using Redis only
if not IS_VERCEL:
    os.makedirs(EMBEDDING_CACHE_DIR, exist_ok=True)
    os.makedirs(ANSWER_CACHE_DIR, exist_ok=True)
    os.makedirs(DOCUMENT_CACHE_DIR, exist_ok=True)

def _get_cache_path(cache_dir: str, key: str) -> str:
    """Get the file path for a cache entry"""
    return os.path.join(cache_dir, f"{key}.pickle")

def _hash_key(key: str) -> str:
    """Create a safe filename from a cache key"""
    return hashlib.md5(key.encode()).hexdigest()

def _serialize_ndarray(arr: np.ndarray) -> bytes:
    """Serialize numpy array for Redis storage"""
    return pickle.dumps(arr)

def _deserialize_ndarray(data: bytes) -> np.ndarray:
    """Deserialize numpy array from Redis storage"""
    return pickle.loads(data)

@lru_cache(maxsize=LRU_MAXSIZE)
def get_cached_embedding(doc_id: str) -> Optional[np.ndarray]:
    """Get document embeddings from cache with environment-aware strategy"""
    key = _hash_key(f"emb:{doc_id}")
    
    # Try Redis first if available
    if USE_REDIS:
        @with_redis_retry()
        def get_from_redis(client=None):
            cached = client.get(f"embedding:{key}")
            if cached:
                return _deserialize_ndarray(cached)
            return None
        
        # Use our retry wrapper
        result = get_from_redis()
        if result is not None:
            return result
    
    # Try memory cache next (for Vercel)
    if IS_VERCEL:
        if key in MEMORY_CACHE["embeddings"]:
            return MEMORY_CACHE["embeddings"][key]
        return None
        
    # Fallback to disk cache for local development
    path = _get_cache_path(EMBEDDING_CACHE_DIR, key)
    if os.path.exists(path):
        try:
            with open(path, 'rb') as f:
                return pickle.load(f)
        except (pickle.PickleError, EOFError):
            os.remove(path)
    
    return None

def cache_embedding(doc_id: str, embeddings: np.ndarray) -> None:
    """Store document embeddings to cache with environment-aware strategy"""
    key = _hash_key(f"emb:{doc_id}")
    
    # Try Redis first if available
    if USE_REDIS:
        @with_redis_retry()
        def store_in_redis(client=None):
            # Store with 24-hour expiration
            client.setex(
                f"embedding:{key}",
                86400,  # 24 hours in seconds
                _serialize_ndarray(embeddings)
            )
            return True
        
        if store_in_redis():
            return
    
    # Use memory cache for Vercel
    if IS_VERCEL:
        MEMORY_CACHE["embeddings"][key] = embeddings
        return
        
    # Fallback to disk cache for local development
    path = _get_cache_path(EMBEDDING_CACHE_DIR, key)
    with open(path, 'wb') as f:
        pickle.dump(embeddings, f)

@lru_cache(maxsize=LRU_MAXSIZE)
def get_cached_answer(doc_url: str, question: str) -> Optional[str]:
    """Get cached answer with environment-aware strategy"""
    key = _hash_key(f"{doc_url}:{question}")
    
    # Try Redis first if available
    if USE_REDIS:
        @with_redis_retry()
        def get_from_redis(client=None):
            cached = client.get(f"answer:{key}")
            if cached:
                return cached.decode('utf-8')
            return None
        
        result = get_from_redis()
        if result:
            return result
    
    # Try memory cache next (for Vercel)
    if IS_VERCEL:
        if key in MEMORY_CACHE["answers"]:
            # Check if cache is fresh (less than 24 hours old)
            entry = MEMORY_CACHE["answers"][key]
            if time.time() - entry["timestamp"] < 86400:
                return entry["value"]
        return None
    
    # Fallback to disk cache for local development
    path = _get_cache_path(ANSWER_CACHE_DIR, key)
    if os.path.exists(path):
        # Check if cache is fresh (less than 24 hours old)
        if time.time() - os.path.getmtime(path) < 86400:
            try:
                with open(path, 'rb') as f:
                    return pickle.load(f)
            except (pickle.PickleError, EOFError):
                os.remove(path)
    return None

def cache_answer(doc_url: str, question: str, answer: str) -> None:
    """Store answer to cache with environment-aware strategy"""
    key = _hash_key(f"{doc_url}:{question}")
    
    # Try Redis first if available
    if USE_REDIS:
        @with_redis_retry()
        def store_in_redis(client=None):
            # Store with 24-hour expiration
            client.setex(
                f"answer:{key}",
                86400,  # 24 hours in seconds
                answer
            )
            return True
        
        if store_in_redis():
            return
    
    # Use memory cache for Vercel
    if IS_VERCEL:
        MEMORY_CACHE["answers"][key] = {
            "value": answer,
            "timestamp": time.time()
        }
        return
        
    # Fallback to disk cache for local development
    path = _get_cache_path(ANSWER_CACHE_DIR, key)
    with open(path, 'wb') as f:
        pickle.dump(answer, f)

def get_cached_document(doc_url: str) -> Optional[Tuple[str, Dict[str, Any]]]:
    """Get cached document text and metadata with environment-aware strategy"""
    key = _hash_key(doc_url)
    
    # Try Redis first if available
    if USE_REDIS:
        @with_redis_retry()
        def get_from_redis(client=None):
            cached = client.get(f"document:{key}")
            if cached:
                return pickle.loads(cached)
            return None
        
        result = get_from_redis()
        if result:
            return result
    
    # Try memory cache next (for Vercel)
    if IS_VERCEL:
        if key in MEMORY_CACHE["documents"]:
            return MEMORY_CACHE["documents"][key]
        return None
    
    # Fallback to disk cache for local development
    path = _get_cache_path(DOCUMENT_CACHE_DIR, key)
    if os.path.exists(path):
        try:
            with open(path, 'rb') as f:
                return pickle.load(f)
        except (pickle.PickleError, EOFError):
            os.remove(path)
    return None

def cache_document(doc_url: str, text: str, meta: Dict[str, Any]) -> None:
    """Cache document text and metadata with environment-aware strategy"""
    key = _hash_key(doc_url)
    data = (text, meta)
    
    # Try Redis first if available
    if USE_REDIS:
        @with_redis_retry()
        def store_in_redis(client=None):
            # Store with 7-day expiration
            client.setex(
                f"document:{key}",
                7 * 86400,  # 7 days in seconds
                pickle.dumps(data)
            )
            return True
        
        if store_in_redis():
            return
    
    # Use memory cache for Vercel
    if IS_VERCEL:
        MEMORY_CACHE["documents"][key] = data
        return
        
    # Fallback to disk cache for local development
    path = _get_cache_path(DOCUMENT_CACHE_DIR, key)
    with open(path, 'wb') as f:
        pickle.dump(data, f)

# Utility function to clear stale cache entries
def clear_stale_cache(max_age_days: int = 7) -> int:
    """Clear cache entries older than max_age_days. Returns number of files removed."""
    if IS_VERCEL:
        logger.info("Skipping clear_stale_cache on Vercel")
        return 0
        
    max_age_seconds = max_age_days * 86400
    now = time.time()
    removed = 0
    
    # Clear Redis cache if available
    if USE_REDIS:
        # We don't need to manually clear Redis as we're using SETEX with expiration
        logger.info("Redis cache uses built-in expiration, skipping manual clear")
    
    # Clear local disk cache
    for cache_dir in [EMBEDDING_CACHE_DIR, ANSWER_CACHE_DIR, DOCUMENT_CACHE_DIR]:
        if os.path.exists(cache_dir):
            for filename in os.listdir(cache_dir):
                filepath = os.path.join(cache_dir, filename)
                if os.path.isfile(filepath) and now - os.path.getmtime(filepath) > max_age_seconds:
                    os.remove(filepath)
                    removed += 1
                
    return removed
