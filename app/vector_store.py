"""
Vector store: Handles embedding, storage, and retrieval using ChromaDB with optimized settings.
"""

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
import numpy as np
import os
import time
from typing import List, Tuple, Dict, Any
import asyncio
from concurrent.futures import ThreadPoolExecutor
from app.utils import hash_str
from app.cache import get_cached_embedding, cache_embedding
import logging
import re
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity

# Optional BM25 for hybrid search if available
try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    # We'll use a simple TF-IDF implementation if BM25 isn't available

logger = logging.getLogger(__name__)

# Constants
EMBED_MODEL = "BAAI/bge-small-en"  # More efficient than e5-small-v2
PERSIST_DIR = os.path.join(os.path.dirname(__file__), "../chromadb_data")
BATCH_SIZE = 64  # Optimal batch size for embedding
DEFAULT_TOP_K = 8  # Increase from 5 to 8 for better coverage and to ensure we catch critical policy clauses

# Create persist directory
os.makedirs(PERSIST_DIR, exist_ok=True)

# Initialize ChromaDB with persistence and optimized settings
chroma_client = chromadb.Client(Settings(
    persist_directory=PERSIST_DIR,
    anonymized_telemetry=False,
    is_persistent=True,
))

# Initialize embedding model
print(f"Loading embedding model: {EMBED_MODEL}...")
model = SentenceTransformer(EMBED_MODEL)
embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)

# Initialize collections
collection = chroma_client.get_or_create_collection(
    name="hackrx_docs",
    embedding_function=embedding_function,
    metadata={"hnsw:space": "cosine"}  # Use HNSW index for faster retrieval
)

def get_or_create_embeddings(doc_url: str, chunks: list[str], refs: list[str] = None):
    """
    Returns (doc_id, embeddings) for the document, using cache if available.
    Optimized with batching for large documents.
    
    Args:
        doc_url: URL of the document
        chunks: List of text chunks
        refs: List of reference information (e.g. page numbers)
    """
    doc_id = hash_str(doc_url)
    
    # Check cache first
    cached = get_cached_embedding(doc_id)
    if cached is not None:
        return doc_id, cached
    
    # Batch process embeddings for efficiency
    all_embeddings = []
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i+BATCH_SIZE]
        batch_embeddings = model.encode(batch, show_progress_bar=True)
        all_embeddings.append(batch_embeddings)
    
    embeddings = np.vstack(all_embeddings) if len(all_embeddings) > 1 else all_embeddings[0]
    
    # Add to ChromaDB in batches
    ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
    
    # Include refs in metadata if available
    metadatas = []
    for i in range(len(chunks)):
        metadata = {
            "doc_id": doc_id, 
            "chunk_idx": i
        }
        if refs and i < len(refs) and refs[i]:
            metadata["page"] = refs[i]
        metadatas.append(metadata)
    
    # Batch upsert for better performance
    for i in range(0, len(chunks), BATCH_SIZE):
        end_idx = min(i + BATCH_SIZE, len(chunks))
        batch_ids = ids[i:end_idx]
        batch_chunks = chunks[i:end_idx]
        batch_embeddings = embeddings[i:end_idx].tolist()
        batch_metadatas = metadatas[i:end_idx]
        
        collection.upsert(
            ids=batch_ids,
            documents=batch_chunks,
            embeddings=batch_embeddings,
            metadatas=batch_metadatas
        )
    
    # Cache for future use
    cache_embedding(doc_id, embeddings)
    return doc_id, embeddings

# New cosine similarity based retrieval
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
    """
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as executor:
        return await loop.run_in_executor(
            executor,
            retrieve_similar_chunks,
            doc_id, query, chunks, refs, embeddings, top_k
        )

def retrieve_similar_chunks(
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
    # Compute query embedding
    query_emb = model.encode([query])[0]
    # Compute cosine similarities
    sims = cosine_similarity(embeddings, query_emb.reshape(1, -1)).flatten()
    # Get top indices
    top_idx = sims.argsort()[-top_k:][::-1]
    # Select chunks and refs
    selected_chunks = [chunks[i] for i in top_idx]
    selected_refs = [refs[i] for i in top_idx]
    return selected_chunks, selected_refs
