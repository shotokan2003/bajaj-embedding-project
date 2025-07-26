"""
Cache module: Handles persistent disk caching and in-memory LRU for embeddings and answers.
Optimized for high-performance retrieval with minimal latency.
"""

import os
import pickle
import hashlib
import time
from functools import lru_cache
from typing import Dict, Any, Optional, Tuple
import numpy as np

# Cache settings
CACHE_DIR = os.path.join(os.path.dirname(__file__), "../cache")
EMBEDDING_CACHE_DIR = os.path.join(CACHE_DIR, "embeddings")
ANSWER_CACHE_DIR = os.path.join(CACHE_DIR, "answers")
DOCUMENT_CACHE_DIR = os.path.join(CACHE_DIR, "documents")
LRU_MAXSIZE = 100  # Number of items to keep in LRU cache

# Create cache directories if they don't exist
os.makedirs(EMBEDDING_CACHE_DIR, exist_ok=True)
os.makedirs(ANSWER_CACHE_DIR, exist_ok=True)
os.makedirs(DOCUMENT_CACHE_DIR, exist_ok=True)

def _get_cache_path(cache_dir: str, key: str) -> str:
    """Get the file path for a cache entry"""
    return os.path.join(cache_dir, f"{key}.pickle")

def _hash_key(key: str) -> str:
    """Create a safe filename from a cache key"""
    return hashlib.md5(key.encode()).hexdigest()

@lru_cache(maxsize=LRU_MAXSIZE)
def get_cached_embedding(doc_id: str) -> Optional[np.ndarray]:
    """Get document embeddings from cache with LRU for frequent docs"""
    key = _hash_key(f"emb:{doc_id}")
    path = _get_cache_path(EMBEDDING_CACHE_DIR, key)
    
    if os.path.exists(path):
        try:
            with open(path, 'rb') as f:
                return pickle.load(f)
        except (pickle.PickleError, EOFError):
            # Handle corrupt cache
            os.remove(path)
    return None

def cache_embedding(doc_id: str, embeddings: np.ndarray) -> None:
    """Store document embeddings to persistent cache"""
    key = _hash_key(f"emb:{doc_id}")
    path = _get_cache_path(EMBEDDING_CACHE_DIR, key)
    
    with open(path, 'wb') as f:
        pickle.dump(embeddings, f)

@lru_cache(maxsize=LRU_MAXSIZE)
def get_cached_answer(doc_url: str, question: str) -> Optional[str]:
    """Get cached answer with LRU for frequent questions"""
    key = _hash_key(f"{doc_url}:{question}")
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
    """Store answer to persistent cache"""
    key = _hash_key(f"{doc_url}:{question}")
    path = _get_cache_path(ANSWER_CACHE_DIR, key)
    
    with open(path, 'wb') as f:
        pickle.dump(answer, f)

def get_cached_document(doc_url: str) -> Optional[Tuple[str, Dict[str, Any]]]:
    """Get cached document text and metadata"""
    key = _hash_key(doc_url)
    path = _get_cache_path(DOCUMENT_CACHE_DIR, key)
    
    if os.path.exists(path):
        try:
            with open(path, 'rb') as f:
                return pickle.load(f)
        except (pickle.PickleError, EOFError):
            os.remove(path)
    return None

def cache_document(doc_url: str, text: str, meta: Dict[str, Any]) -> None:
    """Cache document text and metadata to avoid re-downloading/parsing"""
    key = _hash_key(doc_url)
    path = _get_cache_path(DOCUMENT_CACHE_DIR, key)
    
    with open(path, 'wb') as f:
        pickle.dump((text, meta), f)

# Utility function to clear stale cache entries
def clear_stale_cache(max_age_days: int = 7) -> int:
    """Clear cache entries older than max_age_days. Returns number of files removed."""
    max_age_seconds = max_age_days * 86400
    now = time.time()
    removed = 0
    
    for cache_dir in [EMBEDDING_CACHE_DIR, ANSWER_CACHE_DIR, DOCUMENT_CACHE_DIR]:
        for filename in os.listdir(cache_dir):
            filepath = os.path.join(cache_dir, filename)
            if os.path.isfile(filepath) and now - os.path.getmtime(filepath) > max_age_seconds:
                os.remove(filepath)
                removed += 1
                
    return removed
