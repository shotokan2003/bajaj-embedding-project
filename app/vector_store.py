"""
Vector store: Handles embedding, storage, and retrieval using ChromaDB with cloud embedding APIs.
"""

# First import NumPy patch to ensure compatibility with NumPy 2.0+
from app.numpy_patch import np

# Now we can safely import ChromaDB
import chromadb
from chromadb.config import Settings
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
from app.cloud_embeddings import encode

logger = logging.getLogger(__name__)

# Debug the ChromaDB embedding function interface
logger.info(f"ChromaDB version: {chromadb.__version__}")

# Constants
PERSIST_DIR = os.path.join(os.path.dirname(__file__), "../chromadb_data")
BATCH_SIZE = 16  # Increased batch size for parallel processing
DEFAULT_TOP_K = 8  # Top chunks to retrieve
MAX_CONCURRENT_REQUESTS = 16  # Maximum number of concurrent API requests

# Create persist directory
os.makedirs(PERSIST_DIR, exist_ok=True)

# Initialize ChromaDB with environment-aware settings
# For Vercel deployment, use in-memory storage to avoid filesystem issues
is_vercel = os.environ.get("VERCEL", "0") == "1"

if is_vercel:
    logger.info("Running on Vercel, using in-memory ChromaDB")
    chroma_client = chromadb.Client(Settings(
        anonymized_telemetry=False,
        is_persistent=False,  # In-memory for Vercel
    ))
else:
    logger.info("Running locally, using persistent ChromaDB storage")
    chroma_client = chromadb.Client(Settings(
        persist_directory=PERSIST_DIR,
        anonymized_telemetry=False,
        is_persistent=True,
    ))

# Define a custom embedding function for ChromaDB that uses cloud API
class CloudEmbeddingFunction:
    """
    Custom embedding function that follows the ChromaDB interface
    ChromaDB 0.4.22 requires the parameter name to be exactly 'input'
    """
    def __call__(self, input):
        """
        Generate embeddings for input texts using cloud API.
        
        Args:
            input: List of texts to embed (parameter name must be 'input' for ChromaDB)
            
        Returns:
            List of embeddings as float lists
        """
        try:
            # Log the input type for debugging
            logger.debug(f"CloudEmbeddingFunction called with {len(input)} texts")
            
            # Generate embeddings using our cloud API with optimized batch size
            embeddings = encode(input, batch_size=MAX_CONCURRENT_REQUESTS)
            
            # Ensure result is in the format ChromaDB expects (list of lists of floats)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Error in embedding function: {str(e)}")
            # Return empty embeddings with correct dimensions as fallback
            return [[0.0] * 3072 for _ in range(len(input))]

# Create an instance of the embedding function
embedding_function = CloudEmbeddingFunction()

# Initialize collections
try:
    collection = chroma_client.get_or_create_collection(
        name="hackrx_docs_cloud",  # New collection name to avoid conflicts with previous embeddings
        embedding_function=embedding_function,
        metadata={"hnsw:space": "cosine"}  # Use HNSW index for faster retrieval
    )
    logger.info("ChromaDB collection initialized successfully")
except Exception as e:
    logger.error(f"Error initializing ChromaDB collection: {str(e)}")
    # Fallback to default configuration if there's an error
    try:
        collection = chroma_client.get_or_create_collection(name="hackrx_docs_cloud")
        logger.info("ChromaDB collection initialized with default settings")
    except Exception as e2:
        logger.error(f"Critical error with ChromaDB: {str(e2)}")
        raise

def get_or_create_embeddings(doc_url: str, chunks: list[str], refs: list[str] = None):
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
        return doc_id, cached
    
    # Process embeddings with cloud API
    logger.info(f"Generating embeddings for {len(chunks)} chunks using cloud API")
    embeddings = encode(chunks, batch_size=BATCH_SIZE, show_progress_bar=True)
    
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
    # Compute query embedding using cloud API
    from app.cloud_embeddings import encode
    query_emb = encode([query])[0]
    # Compute cosine similarities
    sims = cosine_similarity(embeddings, query_emb.reshape(1, -1)).flatten()
    # Get top indices
    top_idx = sims.argsort()[-top_k:][::-1]
    # Select chunks and refs
    selected_chunks = [chunks[i] for i in top_idx]
    selected_refs = [refs[i] for i in top_idx]
    return selected_chunks, selected_refs
