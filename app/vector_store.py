"""
Vector store: Handles embedding, storage, and retrieval using Redis cache and cloud embedding APIs.
Optimized for serverless deployment without ChromaDB dependency.
"""

# Import NumPy patch to ensure compatibility with NumPy 2.0+
from app.numpy_patch import np

import os
import time
from typing import List, Tuple, Dict, Any
import asyncio
from concurrent.futures import ThreadPoolExecutor
from app.utils import hash_str
from app.cache import get_cached_embedding, cache_embedding, cache_document
import logging
from app.cloud_embeddings import encode

logger = logging.getLogger(__name__)

# Constants
BATCH_SIZE = 32  # Batch size for parallel processing
DEFAULT_TOP_K = 8  # Top chunks to retrieve
MAX_CONCURRENT_REQUESTS = 32  # Maximum number of concurrent API requests

logger.info("Vector store initialized with Redis-only storage")

def cosine_similarity_np(embeddings: np.ndarray, query_emb: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between embeddings and query using NumPy only.
    Much faster and lighter than scikit-learn for this simple operation.
    
    Args:
        embeddings: Matrix of document embeddings (n_docs, n_features)
        query_emb: Query embedding vector (n_features,)
    
    Returns:
        Array of cosine similarities (n_docs,)
    """
    # Normalize embeddings and query
    embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    query_norm = query_emb / np.linalg.norm(query_emb)
    
    # Compute cosine similarity as dot product of normalized vectors
    return np.dot(embeddings_norm, query_norm)

async def get_or_create_embeddings(doc_url: str, chunks: list[str], refs: list[str] = None):
    """
    Returns (doc_id, embeddings) for the document, using cache if available.
    Uses cloud embedding API with appropriate batching.
    
    Args:
        doc_url: URL of the document
        chunks: List of text chunks
        refs: List of reference information (e.g. page numbers)
    """
    doc_id = hash_str(doc_url)
    
    # Check cache first
    cached = get_cached_embedding(doc_id)
    if cached is not None:
        logger.info(f"Using cached embeddings for document {doc_id}")
        return doc_id, cached
    
    # Process embeddings with cloud API (run in thread pool since encode is CPU-bound)
    logger.info(f"Generating embeddings for {len(chunks)} chunks using cloud API")
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_REQUESTS) as executor:
        embeddings = await loop.run_in_executor(
            executor, 
            lambda: encode(chunks, batch_size=BATCH_SIZE, show_progress_bar=True)
        )
    
    # Cache embeddings and document chunks for future use (run in task to avoid blocking)
    # We don't need to await this as it's not critical for the response
    asyncio.create_task(async_cache_operations(doc_id, doc_url, embeddings, chunks, refs))
    
    logger.info(f"Cached embeddings and document data for {doc_id}")
    return doc_id, embeddings

async def async_cache_operations(doc_id: str, doc_url: str, embeddings: np.ndarray, chunks: list[str], refs: list[str] = None):
    """Helper function to handle cache operations asynchronously"""
    # Cache embeddings
    cache_embedding(doc_id, embeddings)
    
    # Store document chunks and refs for retrieval
    doc_data = {
        "chunks": chunks,
        "refs": refs or []
    }
    cache_document(doc_url, doc_data, {"processed": True})

# Cosine similarity based retrieval - now directly using the async version
async def retrieve_similar_chunks_async(
    doc_id: str,
    query: str,
    chunks: list[str],
    refs: list[str],
    embeddings: np.ndarray,
    top_k: int = DEFAULT_TOP_K
) -> tuple[list[str], list[str]]:
    """
    Async cosine-based retrieval using precomputed embeddings.
    This function is now a simple wrapper around the async retrieve_similar_chunks.
    """
    return await retrieve_similar_chunks(doc_id, query, chunks, refs, embeddings, top_k)

async def retrieve_similar_chunks(
    doc_id: str,
    query: str,
    chunks: list[str],
    refs: list[str],
    embeddings: np.ndarray,
    top_k: int = DEFAULT_TOP_K
) -> tuple[list[str], list[str]]:
    """
    Compute cosine similarity between query embedding and all chunk embeddings,
    return top_k most similar chunks and their refs.
    """
    # Compute query embedding using cloud API (run in thread pool since encode is CPU-bound)
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as executor:
        query_emb = await loop.run_in_executor(executor, lambda: encode([query])[0])
    
    # Compute cosine similarities using our lightweight NumPy implementation
    # This is fast enough to not need thread pool
    sims = cosine_similarity_np(embeddings, query_emb)
    
    # Get top indices
    top_idx = sims.argsort()[-top_k:][::-1]
    
    # Select chunks and refs
    selected_chunks = [chunks[i] for i in top_idx]
    selected_refs = [refs[i] for i in top_idx]
    return selected_chunks, selected_refs
