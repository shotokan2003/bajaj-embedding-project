"""
Cache module STUB: All caching has been removed. 
This file provides stub functions that do nothing but maintain API compatibility.
Direct document parsing and embedding is used instead of any caching.
"""

import os
import logging
import numpy as np
from typing import Dict, Any, Optional, Tuple, Union
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Fix for NumPy 2.0+: ensure we use float64 instead of float_
if hasattr(np, 'float_'):
    np.float64 = np.float_

# Caching is disabled
USE_REDIS = False
IS_VERCEL = False

# Empty cache directories (for API compatibility)
CACHE_DIR = os.path.join(os.path.dirname(__file__), "../cache")
EMBEDDING_CACHE_DIR = os.path.join(CACHE_DIR, "embeddings")
ANSWER_CACHE_DIR = os.path.join(CACHE_DIR, "answers")
DOCUMENT_CACHE_DIR = os.path.join(CACHE_DIR, "documents")

# Log that caching is disabled
logger.info("All caching disabled - using direct document parsing and embedding")

def _hash_key(key: str) -> str:
    """Create a hash of a key (for API compatibility)"""
    return hashlib.md5(key.encode()).hexdigest()

def get_cached_embedding(doc_id: str) -> Optional[np.ndarray]:
    """Stub function - caching disabled, always returns None"""
    logger.debug(f"Cache disabled: not retrieving embeddings for {doc_id}")
    return None

def cache_embedding(doc_id: str, embeddings: np.ndarray) -> None:
    """Stub function - caching disabled, does nothing"""
    logger.debug(f"Cache disabled: not storing embeddings for {doc_id}")
    pass

def get_cached_answer(doc_url: str, question: str) -> Optional[str]:
    """Stub function - caching disabled, always returns None"""
    logger.debug(f"Cache disabled: not retrieving answer for {question}")
    return None

def cache_answer(doc_url: str, question: str, answer: str) -> None:
    """Stub function - caching disabled, does nothing"""
    logger.debug(f"Cache disabled: not storing answer for {question}")
    pass

def get_cached_document(doc_url: str) -> Optional[Tuple[str, Dict[str, Any]]]:
    """Stub function - caching disabled, always returns None"""
    logger.debug(f"Cache disabled: not retrieving document from {doc_url}")
    return None

def cache_document(doc_url: str, text: str, meta: Dict[str, Any]) -> None:
    """Stub function - caching disabled, does nothing"""
    logger.debug(f"Cache disabled: not storing document from {doc_url}")
    pass

def clear_stale_cache(max_age_days: int = 7) -> int:
    """Stub function - caching disabled, returns 0 (no files removed)"""
    logger.debug("Cache disabled: not clearing cache")
    return 0
