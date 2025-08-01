"""
Google Generative AI (Gemini) embedding client for generating text embeddings.
This module replaces the SentenceTransformer local model with Google's Gemini embedding model.
"""

import os
import numpy as np
from typing import List, Dict, Any, Union
from dotenv import load_dotenv
import logging
from time import sleep
from tenacity import retry, stop_after_attempt, wait_exponential
import google.generativeai as genai

# Configure logging
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# API Configuration
API_KEY = os.getenv("GOOGLE_API_KEY")
EMBEDDING_MODEL = "gemini-embedding-001"  # Google's Gemini embedding model
EMBEDDING_DIMENSION = 768  # Gemini embedding dimension

# Initialize the Google AI client
genai.configure(api_key=API_KEY)

class EmbeddingError(Exception):
    """Custom exception for embedding API errors."""
    pass

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    reraise=True
)
def get_embedding(text: str) -> np.ndarray:
    """
    Get embedding for a single text using Google's Gemini embedding model.
    
    Args:
        text: The text to embed
        
    Returns:
        A numpy array containing the embedding vector
    """
    if not API_KEY:
        raise EmbeddingError("Google API key not found in environment variables.")
    
    # Check for empty text
    if not text or not text.strip():
        logger.warning("Empty text provided for embedding, returning zero vector")
        return np.zeros(EMBEDDING_DIMENSION)
    
    # Ensure text is not too long (max tokens for Gemini model)
    if len(text.split()) > 3000:
        logger.warning("Text too long, truncating to 3000 words")
        text = " ".join(text.split()[:3000])
    
    try:
        # Get embeddings using the Google GenerativeAI API
        result = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=text
        )
        
        # Extract the embedding values
        embedding = result["embedding"]
        
        if not embedding:
            logger.error("Failed to get embedding: empty values returned")
            raise EmbeddingError("No embedding values returned")
        
        return np.array(embedding)
    
    except Exception as e:
        logger.error(f"API request error: {str(e)}")
        raise EmbeddingError(f"API request failed: {str(e)}")

import hashlib
import concurrent.futures
import functools
import threading

# Thread-local cache for embeddings to reduce API calls
_local_cache = threading.local()
_cache_lock = threading.Lock()
_embedding_cache = {}  # Global cache

def _hash_text(text: str) -> str:
    """Create a hash for the text to use as a cache key"""
    return hashlib.md5(text.encode('utf-8')).hexdigest()

def get_embedding_cached(text: str) -> np.ndarray:
    """Get embedding with caching to reduce API calls"""
    # Generate hash for cache key
    text_hash = _hash_text(text)
    
    # Check if we have it in our cache
    with _cache_lock:
        if text_hash in _embedding_cache:
            return _embedding_cache[text_hash]
    
    # If not in cache, generate and store
    embedding = get_embedding(text)
    
    with _cache_lock:
        _embedding_cache[text_hash] = embedding
    
    return embedding

def encode(texts: List[str], batch_size: int = 16, show_progress_bar: bool = False) -> np.ndarray:
    """
    Encode a list of texts to get embeddings with parallel processing.
    This method mimics the interface of SentenceTransformer.encode().
    
    Args:
        texts: List of texts to encode
        batch_size: Number of texts to process in parallel
        show_progress_bar: Whether to show a progress bar (ignored, for compatibility only)
        
    Returns:
        A numpy array of embeddings
    """
    if not texts:
        return np.array([])
    
    # Optimize batch size based on number of texts
    if len(texts) <= 4:
        # For small batches, process sequentially
        embeddings = [get_embedding_cached(text) for text in texts]
        return np.array(embeddings)
    
    # Use thread pool for parallel processing
    num_workers = min(batch_size, 32)  # Cap at 32 threads
    
    # Process texts in parallel using thread pool
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        if show_progress_bar:
            logger.info(f"Processing {len(texts)} texts with {num_workers} workers")
        
        embeddings = list(executor.map(get_embedding_cached, texts))
        
        if show_progress_bar:
            logger.info(f"Completed embedding {len(texts)} texts")
    
    return np.array(embeddings)
