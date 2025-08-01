"""
OpenAI embedding API client for generating text embeddings.
This module replaces the SentenceTransformer local model with OpenAI's embedding API.
"""

import os
import requests
import numpy as np
from typing import List, Dict, Any, Union
from dotenv import load_dotenv
import json
import logging
from time import sleep
from tenacity import retry, stop_after_attempt, wait_exponential

# Configure logging
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# API Configuration
API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = "text-embedding-3-small"  # Ada 3 - smaller, faster, cheaper
EMBEDDING_URL = "https://api.openai.com/v1/embeddings"
EMBEDDING_DIMENSION = 1536  # text-embedding-3-small model dimension

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
    Get embedding for a single text using OpenAI's embedding API.
    
    Args:
        text: The text to embed
        
    Returns:
        A numpy array containing the embedding vector
    """
    if not API_KEY:
        raise EmbeddingError("OpenAI API key not found in environment variables.")
    
    # Check for empty text
    if not text or not text.strip():
        logger.warning("Empty text provided for embedding, returning zero vector")
        return np.zeros(EMBEDDING_DIMENSION)
    
    # Ensure text is not too long (max ~8000 tokens for OpenAI embeddings)
    if len(text.split()) > 3000:
        logger.warning("Text too long, truncating to 3000 words")
        text = " ".join(text.split()[:3000])
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    
    payload = {
        "input": text,
        "model": EMBEDDING_MODEL
    }
    
    try:
        response = requests.post(EMBEDDING_URL, headers=headers, json=payload)
        response.raise_for_status()  # Raise exception for error status codes
        
        data = response.json()
        embedding = data.get("data", [{}])[0].get("embedding")
        
        if not embedding:
            logger.error(f"Failed to get embedding: {data}")
            raise EmbeddingError(f"No embedding returned: {data}")
        
        return np.array(embedding)
    
    except requests.exceptions.RequestException as e:
        logger.error(f"API request error: {str(e)}")
        if response and hasattr(response, 'text'):
            logger.error(f"Response: {response.text}")
        raise EmbeddingError(f"API request failed: {str(e)}")

def encode(texts: List[str], batch_size: int = 8, show_progress_bar: bool = False) -> np.ndarray:
    """
    Encode a list of texts to get embeddings.
    This method mimics the interface of SentenceTransformer.encode().
    
    Args:
        texts: List of texts to encode
        batch_size: Number of texts to process in parallel
        show_progress_bar: Whether to show a progress bar (ignored, for compatibility only)
        
    Returns:
        A numpy array of embeddings
    """
    embeddings = []
    
    # Process in batches
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        batch_embeddings = []
        
        for text in batch:
            embedding = get_embedding(text)
            batch_embeddings.append(embedding)
            # Small delay to avoid rate limiting
            sleep(0.1)
        
        embeddings.extend(batch_embeddings)
        
        if i % 50 == 0 and i > 0 and show_progress_bar:
            logger.info(f"Processed {i}/{len(texts)} texts")
    
    return np.array(embeddings)
