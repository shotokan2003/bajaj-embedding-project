"""
Google Generative AI (Gemini) embedding client for generating text embeddings.
This module replaces the SentenceTransformer local model with Google's Gemini embedding model.
"""

import os
import numpy as np
from typing import List, Dict, Any, Union
from dotenv import load_dotenv
import logging
import time
from tenacity import retry, stop_after_attempt, wait_exponential
import google.generativeai as genai
import concurrent.futures

# Load configuration
try:
    from app.config import config
except ImportError:
    # Fallback if config is not available
    class DefaultConfig:
        EMBEDDING_BATCH_SIZE = 24
        RATE_LIMIT_DELAY = 0.1
    config = DefaultConfig()

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
    if len(text.split()) > 1500:  # Reduced from 3000 for better performance
        logger.warning("Text too long, truncating to 1500 words")
        text = " ".join(text.split()[:1500])
    
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

def encode(texts: List[str], batch_size: int = None, show_progress_bar: bool = False) -> np.ndarray:
    """
    Encode a list of texts with optimized batch processing and smart deduplication.
    Enhanced for better performance and accuracy.
    
    Args:
        texts: List of texts to encode
        batch_size: Optimal batch size (uses config default if None)
        show_progress_bar: Whether to show a progress bar
        
    Returns:
        A numpy array of embeddings
    """
    if batch_size is None:
        batch_size = config.EMBEDDING_BATCH_SIZE
        
    if not texts:
        return np.array([])
    
    # Smart deduplication with normalization
    normalized_texts = []
    text_to_normalized = {}
    
    for text in texts:
        # Normalize text for deduplication (remove extra spaces, case insensitive)
        normalized = ' '.join(text.lower().strip().split())
        normalized_texts.append(normalized)
        if normalized not in text_to_normalized:
            text_to_normalized[normalized] = text
    
    # Get unique texts preserving original formatting
    unique_normalized = list(text_to_normalized.keys())
    unique_texts = [text_to_normalized[norm] for norm in unique_normalized]
    
    # Create mapping for reconstruction
    idx_to_unique_idx = {
        i: unique_normalized.index(normalized_texts[i]) 
        for i in range(len(texts))
    }
    
    # Ultra-fast processing strategy - minimal delays
    if len(unique_texts) <= 8:
        # Sequential processing for small batches - fastest for <8 texts
        unique_embeddings = []
        for text in unique_texts:
            try:
                embedding = get_embedding(text)
                unique_embeddings.append(embedding)
                time.sleep(0.01)  # Minimal delay
            except Exception as e:
                logger.error(f"Error encoding text: {str(e)}")
                unique_embeddings.append(np.zeros(EMBEDDING_DIMENSION))
    else:
        # Aggressive parallel processing for larger batches
        max_workers = 8  # High worker count for speed
        chunk_size = 3   # Very small chunks for fastest processing
        unique_embeddings = []
        
        for i in range(0, len(unique_texts), chunk_size):
            chunk_texts = unique_texts[i:i + chunk_size]
            
            # Fast parallel processing
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                chunk_embeddings = list(executor.map(safe_get_embedding, chunk_texts))
            
            unique_embeddings.extend(chunk_embeddings)
            
            # Minimal delay for maximum speed
            if i + chunk_size < len(unique_texts):
                time.sleep(0.01)
            
            # Report only completion
            if show_progress_bar and i + chunk_size >= len(unique_texts):
                logger.info(f"Embedding progress: 100%")
            
            unique_embeddings.extend(chunk_embeddings)
            
            # Reduced delay between chunks for faster processing
            if i + chunk_size < len(unique_texts):
                time.sleep(0.05)  # Reduced from config.RATE_LIMIT_DELAY
            
            # Progress reporting - less frequent
            if show_progress_bar and len(unique_texts) > 20:
                progress = min(100, (i + chunk_size) * 100 // len(unique_texts))
                if progress % 50 == 0:  # Report every 50% instead of 25%
                    logger.info(f"Embedding progress: {progress}%")
    
    # Map embeddings back to original order
    final_embeddings = [
        unique_embeddings[idx_to_unique_idx[i]] 
        for i in range(len(texts))
    ]
    
    return np.array(final_embeddings)

def safe_get_embedding(text: str) -> np.ndarray:
    """Wrapper for get_embedding with error handling"""
    try:
        return get_embedding(text)
    except Exception as e:
        logger.error(f"Error encoding text: {str(e)}")
        return np.zeros(EMBEDDING_DIMENSION)
