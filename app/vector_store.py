"""
Vector store: Handles embedding and vector-based retrieval using cloud embedding APIs.
Optimized for serverless deployment with direct processing (no database or caching).
"""

# Import NumPy patch to ensure compatibility with NumPy 2.0+
from app.numpy_patch import np

import os
import time
from typing import List, Tuple, Dict, Any
import asyncio
from concurrent.futures import ThreadPoolExecutor
from app.utils import hash_str
import logging
from app.cloud_embeddings import encode
from app.config import config
from app.performance import monitor_performance

logger = logging.getLogger(__name__)

# Constants from configuration
BATCH_SIZE = config.EMBEDDING_BATCH_SIZE
DEFAULT_TOP_K = config.DEFAULT_TOP_K
MAX_CONCURRENT_REQUESTS = config.MAX_CONCURRENT_EMBEDDINGS
RERANK_TOP_K = config.RERANK_TOP_K

logger.info("Vector store initialized with direct API processing (no caching)")

def hybrid_similarity_score(
    semantic_sim: float, 
    keyword_score: float, 
    position_score: float,
    chunk_type_score: float,
    semantic_weight: float = config.SEMANTIC_WEIGHT,
    keyword_weight: float = config.KEYWORD_WEIGHT,
    position_weight: float = config.POSITION_WEIGHT,
    type_weight: float = config.TYPE_WEIGHT
) -> float:
    """
    Combine multiple similarity signals for better ranking.
    
    Args:
        semantic_sim: Cosine similarity score
        keyword_score: Keyword overlap score
        position_score: Position-based score (early chunks might be more important)
        chunk_type_score: Score based on chunk type (policy sections get higher scores)
        semantic_weight: Weight for semantic similarity
        keyword_weight: Weight for keyword matching
        position_weight: Weight for position in document
        type_weight: Weight for chunk type
    
    Returns:
        Combined hybrid score
    """
    # Handle None values with safe defaults
    semantic_sim = semantic_sim if semantic_sim is not None else 0.0
    keyword_score = keyword_score if keyword_score is not None else 0.0
    position_score = position_score if position_score is not None else 0.5
    chunk_type_score = chunk_type_score if chunk_type_score is not None else 0.5
    
    return (
        semantic_sim * semantic_weight +
        keyword_score * keyword_weight +
        position_score * position_weight +
        chunk_type_score * type_weight
    )

def compute_keyword_score(query: str, chunk: str) -> float:
    """Compute keyword overlap score between query and chunk"""
    try:
        if not query or not chunk:
            return 0.0
            
        query_words = set(query.lower().split())
        chunk_words = set(chunk.lower().split())
        
        if not query_words:
            return 0.0
        
        # Exact word matches
        exact_matches = len(query_words & chunk_words)
        
        # Partial matches (substring matching for compound words)
        partial_matches = 0
        for q_word in query_words:
            for c_word in chunk_words:
                if len(q_word) > 3 and (q_word in c_word or c_word in q_word):
                    partial_matches += 0.5
        
        # Normalize by query length
        return (exact_matches + partial_matches) / len(query_words)
    except Exception as e:
        logger.warning(f"Error computing keyword score: {e}")
        return 0.0

def compute_position_score(chunk_index: int, total_chunks: int) -> float:
    """Compute position-based score (earlier chunks get slightly higher scores)"""
    try:
        if total_chunks <= 1 or chunk_index < 0:
            return 1.0
        
        # Ensure chunk_index is within bounds
        chunk_index = min(chunk_index, total_chunks - 1)
        
        # Linear decay from 1.0 to 0.5
        return 1.0 - 0.5 * (chunk_index / (total_chunks - 1))
    except Exception as e:
        logger.warning(f"Error computing position score: {e}")
        return 0.5

def compute_chunk_type_score(chunk: str) -> float:
    """Compute score based on chunk type and markers"""
    try:
        if not chunk:
            return 0.5
            
        chunk_lower = chunk.lower()
        
        # Policy sections get highest score
        if '[policy section:' in chunk_lower:
            return 1.0
        
        # Topic summaries get high score
        if '[topic summary:' in chunk_lower:
            return 0.9
        
        # Table content gets medium-high score
        if '[table start]' in chunk_lower:
            return 0.8
        
        # Headers get medium score
        if '===' in chunk or '---' in chunk:
            return 0.7
        
        # Regular chunks
        return 0.5
    except Exception as e:
        logger.warning(f"Error computing chunk type score: {e}")
        return 0.5
    
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

@monitor_performance("embedding_generation")
async def get_or_create_embeddings(doc_url: str, chunks: list[str], refs: list[str] = None):
    """
    Returns (doc_id, embeddings) for the document.
    Uses cloud embedding API with appropriate batching.
    
    Args:
        doc_url: URL of the document
        chunks: List of text chunks
        refs: List of reference information (e.g. page numbers)
    """
    doc_id = hash_str(doc_url)
    
    # Process embeddings with cloud API - optimized for speed
    logger.info(f"Generating embeddings for {len(chunks)} chunks using cloud API")
    loop = asyncio.get_event_loop()
    
    # Use smaller batch size for faster processing
    fast_batch_size = min(BATCH_SIZE, 16)  # Cap at 16 for speed
    with ThreadPoolExecutor(max_workers=8) as executor:  # Reduced workers
        embeddings = await loop.run_in_executor(
            executor, 
            lambda: encode(chunks, batch_size=fast_batch_size, show_progress_bar=True)
        )
    
    logger.info(f"Generated embeddings for document {doc_id}")
    return doc_id, embeddings

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

@monitor_performance("chunk_retrieval")
async def retrieve_similar_chunks(
    doc_id: str,
    query: str,
    chunks: list[str],
    refs: list[str],
    embeddings: np.ndarray,
    top_k: int = 6  # Balanced for accuracy
) -> tuple[list[str], list[str]]:
    """
    Balanced retrieval optimizing both speed and accuracy.
    Uses semantic similarity with smart keyword boosting.
    """
    # Fast query embedding
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor(max_workers=2) as executor:
        query_emb = await loop.run_in_executor(executor, lambda: encode([query])[0])
    
    # Enhanced semantic similarity calculation
    semantic_sims = cosine_similarity_np(embeddings, query_emb)
    
    # Smart keyword analysis for accuracy
    query_words = set(query.lower().split())
    # Extract important terms (exclude common words)
    stop_words = {'is', 'are', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
    important_query_words = query_words - stop_words
    
    # Enhanced keyword scoring
    keyword_scores = []
    for chunk in chunks:
        chunk_words = set(chunk.lower().split())
        # Calculate keyword overlap with emphasis on important terms
        important_matches = len(important_query_words & chunk_words)
        total_matches = len(query_words & chunk_words)
        
        # Weighted keyword score (important words count more)
        keyword_score = (important_matches * 0.3) + (total_matches * 0.1)
        keyword_scores.append(keyword_score)
    
    # Combine scores with optimal weighting for accuracy
    final_scores = []
    for i in range(len(chunks)):
        # Balance semantic and keyword matching
        combined_score = (semantic_sims[i] * 0.8) + (keyword_scores[i] * 0.2)
        final_scores.append(combined_score)
    
    # Get top results with better ranking
    top_indices = np.argsort(final_scores)[-top_k:][::-1]
    
    selected_chunks = [chunks[i] for i in top_indices]
    selected_refs = [refs[i] if i < len(refs) else "" for i in top_indices]
    
    return selected_chunks, selected_refs

async def rerank_chunks(
    query: str, 
    candidate_chunks: list[str], 
    candidate_indices: list[int],
    hybrid_scores: list[float]
) -> list[int]:
    """
    Re-rank candidates based on query-specific criteria.
    Applies additional heuristics for better relevance.
    """
    query_lower = query.lower()
    
    # Define query type patterns
    query_patterns = {
        'coverage': ['covered', 'cover', 'coverage', 'include', 'eligible'],
        'waiting': ['waiting', 'wait', 'period', 'duration', 'months', 'years'],
        'grace': ['grace', 'period', 'due', 'payment', 'premium'],
        'amount': ['amount', 'limit', 'rupees', 'cost', 'price', 'charges'],
        'exclusion': ['not covered', 'excluded', 'exclusion', 'does not cover'],
        'procedure': ['procedure', 'treatment', 'surgery', 'operation', 'therapy']
    }
    
    # Score each candidate based on query type relevance
    query_type_scores = []
    for i, chunk in enumerate(candidate_chunks):
        chunk_lower = chunk.lower()
        type_score = 0
        
        for pattern_type, keywords in query_patterns.items():
            query_matches = sum(1 for kw in keywords if kw in query_lower)
            chunk_matches = sum(1 for kw in keywords if kw in chunk_lower)
            
            if query_matches > 0 and chunk_matches > 0:
                type_score += (query_matches * chunk_matches) / len(keywords)
        
        # Boost scores for chunks with exact phrase matches
        query_words = query_lower.split()
        if len(query_words) > 1:
            for j in range(len(query_words) - 1):
                phrase = ' '.join(query_words[j:j+2])
                if phrase in chunk_lower:
                    type_score += 0.5
        
        query_type_scores.append(type_score)
    
    # Combine original hybrid scores with query type scores
    final_scores = [
        hybrid_scores[candidate_indices[i]] * 0.7 + query_type_scores[i] * 0.3
        for i in range(len(candidate_chunks))
    ]
    
    # Return indices sorted by final scores
    sorted_positions = sorted(range(len(final_scores)), key=lambda x: final_scores[x], reverse=True)
    return [candidate_indices[pos] for pos in sorted_positions]
