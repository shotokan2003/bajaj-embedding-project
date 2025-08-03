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

import concurrent.futures
import functools

# No caching - direct API calls only

def encode(texts: List[str], batch_size: int = 16, show_progress_bar: bool = False) -> np.ndarray:
    """
    Encode a list of texts to get embeddings with parallel processing.
    This method mimics the interface of SentenceTransformer.encode().
    Optimized for performance with chunked batching and thread pool.
    
    Args:
        texts: List of texts to encode
        batch_size: Number of texts to process in parallel
        show_progress_bar: Whether to show a progress bar (ignored, for compatibility only)
        
    Returns:
        A numpy array of embeddings
    """
    if not texts:
        return np.array([])
    
    # Deduplicate texts to avoid redundant API calls within the same batch
    unique_texts = list(set(texts))
    text_to_idx = {text: i for i, text in enumerate(texts)}
    idx_to_unique_idx = {i: unique_texts.index(text) for i, text in enumerate(texts)}
    
    # For very small batches, process sequentially to avoid thread overhead
    if len(unique_texts) <= 4:
        unique_embeddings = [get_embedding(text) for text in unique_texts]
    else:
        # Use thread pool for parallel processing with optimal worker count
        # More workers for larger batches, but cap based on CPU count
        num_workers = min(max(batch_size, 8), os.cpu_count() * 2 or 16)
        
        # Process in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            unique_embeddings = list(executor.map(get_embedding, unique_texts))
    
    # Map back to original order
    final_embeddings = [unique_embeddings[idx_to_unique_idx[i]] for i in range(len(texts))]
    return np.array(final_embeddings)
        
        # This code block is unreachable after our updates and should be removed
    
    # Map unique embeddings back to original texts
    final_embeddings = [unique_embeddings[idx_to_unique_idx[i]] for i in range(len(texts))]
    return np.array(final_embeddings)
