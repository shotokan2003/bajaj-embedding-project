"""
Vector store: Handles embedding, storage, and retrieval using ChromaDB with optimized settings.
"""

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
import numpy as np
import os
from typing import List, Tuple, Dict, Any
import asyncio
from concurrent.futures import ThreadPoolExecutor
from app.utils import hash_str
from app.cache import get_cached_embedding, cache_embedding

# Constants
EMBED_MODEL = "BAAI/bge-small-en"  # More efficient than e5-small-v2
PERSIST_DIR = os.path.join(os.path.dirname(__file__), "../chromadb_data")
BATCH_SIZE = 64  # Optimal batch size for embedding

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

def get_or_create_embeddings(doc_url: str, chunks: list[str]):
    """
    Returns (doc_id, embeddings) for the document, using cache if available.
    Optimized with batching for large documents.
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
        batch_embeddings = model.encode(batch, show_progress_bar=False)
        all_embeddings.append(batch_embeddings)
    
    embeddings = np.vstack(all_embeddings) if len(all_embeddings) > 1 else all_embeddings[0]
    
    # Add to ChromaDB in batches
    ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
    metadatas = [{"doc_id": doc_id, "chunk_idx": i} for i in range(len(chunks))]
    
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

async def retrieve_similar_chunks_async(doc_id: str, query: str, chunks: list[str], refs: list[str], top_k: int = 3):
    """
    Async version of retrieve_similar_chunks for parallel processing
    """
    # Run in thread pool since ChromaDB operations are blocking
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as executor:
        return await loop.run_in_executor(
            executor,
            retrieve_similar_chunks,
            doc_id, query, chunks, refs, top_k
        )

def retrieve_similar_chunks(doc_id: str, query: str, chunks: list[str], refs: list[str], top_k: int = 3):
    """
    Embeds the query, retrieves top-k similar chunks from ChromaDB.
    Returns (chunks, refs).
    """
    # Optimized query with filtering by doc_id
    query_emb = model.encode([query])
    results = collection.query(
        query_embeddings=query_emb.tolist(),
        n_results=top_k,
        where={"doc_id": doc_id},  # Filter by doc_id for faster retrieval
        include=["documents", "metadatas", "distances"]
    )
    
    # Get indices from IDs (needed for refs lookup)
    indices = [int(idx.split("_")[-1]) for idx in results["ids"][0]]
    
    # Extract the relevant chunks and refs
    top_chunks = [chunks[idx] for idx in indices]
    top_refs = [refs[idx] for idx in indices]
    
    return top_chunks, top_refs
