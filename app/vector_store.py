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

async def retrieve_similar_chunks_async(doc_id: str, query: str, chunks: list[str], refs: list[str], top_k: int = DEFAULT_TOP_K):
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

def retrieve_similar_chunks(doc_id: str, query: str, chunks: list[str], refs: list[str], top_k: int = DEFAULT_TOP_K):
    """
    Implements a true hybrid search combining vector similarity and lexical search.
    Returns (chunks, refs) optimized for relevance and diversity.
    
    Improvements:
    1. BM25 keyword matching for exact term presence
    2. Vector similarity for semantic understanding
    3. Balanced hybrid score combining both approaches
    4. Diversity boosting for different pages/sections
    5. Special handling for insurance policy terms
    """
    # Dictionary of insurance policy term expansions for query enhancement
    expansion_dict = {
        "grace period": ["grace period", "premium payment", "due date", "renewal", "continuity", "days grace", 
                         "period of grace", "renewal premium", "policy expiry", "policy termination", 
                         "lapsed policy", "policy continuation", "break in policy"],
        "waiting period": ["waiting period", "pre-existing", "disease", "coverage", "exclusion", 
                          "cooling period", "months waiting", "years waiting", "condition"],
        "maternity": ["maternity", "childbirth", "delivery", "pregnancy", "cesarean", "female insured", 
                     "baby", "newborn", "labor", "delivery expenses"],
        "cataract": ["cataract", "surgery", "eye", "lens", "vision", "ophthalmology", "eyesight", 
                    "ocular", "ophthalmic procedure"],
        "organ donor": ["organ donor", "transplantation", "kidney", "liver", "donation", "recipient", 
                       "transplant", "human organs", "donor expenses"],
        "no claim": ["no claim", "discount", "NCD", "bonus", "premium", "renewal", "claim free", 
                    "reward", "benefit", "discount on premium"],
        "health check": ["health check", "preventive", "wellness", "benefit", "check-up", "screening", 
                        "diagnostic", "annual", "free check", "medical examination"],
        "hospital": ["hospital", "definition", "nursing home", "beds", "medical staff", "healthcare facility", 
                    "inpatient care", "medical practitioner", "registration", "admission"],
        "ayush": ["ayush", "ayurveda", "yoga", "naturopathy", "unani", "siddha", "homeopathy", 
                 "alternative medicine", "traditional medicine", "holistic treatment"],
        "room rent": ["room rent", "ICU", "charges", "sub-limit", "plan", "single private room", 
                     "accommodation", "hospital room", "daily limit", "boarding expenses"]
    }
    
    # Add general insurance terms that might be relevant for any query
    general_insurance_terms = [
        "policy period", "insured", "coverage", "exclusions", "policy terms", "conditions",
        "sum insured", "deductible", "co-payment", "premium", "claim", "benefit",
        "renewal", "endorsement", "rider", "grace period", "limit", "policy document",
        "insurance contract", "policy holder", "policy number", "health insurance"
    ]
    
    # Create a list of critical terms to search for explicitly
    critical_terms = []
    query_lower = query.lower()
    
    # Match query to domain terms and add relevant expansions
    matched_any = False
    for key, terms in expansion_dict.items():
        if any(term in query_lower for term in terms[:2]):
            critical_terms.extend(terms[:5])  # Add top 5 most relevant terms
            matched_any = True
    
    # Add some general terms if no specific match was found
    if not matched_any:
        critical_terms.extend(general_insurance_terms[:3])
    
    # Enhanced query for vector search
    enhanced_query = query + " " + " ".join(critical_terms[:5])
    
    # Generate query embedding for semantic search
    query_emb = model.encode([enhanced_query])    # Step 1: Perform vector-based semantic search
    # Get more results than needed to ensure diversity and coverage
    vector_k = min(top_k * 2, len(chunks))  # Get twice as many results initially
    
    # Perform vector search through ChromaDB
    results = collection.query(
        query_embeddings=query_emb.tolist(),
        n_results=vector_k,
        where={"doc_id": doc_id},
        include=["documents", "metadatas", "distances"]
    )
    
    if not results["ids"][0]:
        return [], []
        
    # Get indices and distances from vector search
    result_ids = results["ids"][0]
    result_distances = results["distances"][0] if "distances" in results else [1.0] * len(result_ids)
    result_metadatas = results["metadatas"][0]
    
    # Step 2: Perform keyword-based search (BM25 or TF-IDF)
    # Prepare for lexical search
    chunk_indices = []
    corpus = []
    
    # Process chunks for keyword search
    for i, chunk in enumerate(chunks):
        chunk_indices.append(i)
        # Tokenize: split into words and remove punctuation
        tokens = re.findall(r'\b\w+\b', chunk.lower())
        corpus.append(tokens)
    
    # Use BM25 if available, otherwise a simplified TF-IDF approach
    if BM25_AVAILABLE:
        # Initialize BM25 with our corpus
        bm25 = BM25Okapi(corpus)
        
        # Tokenize query same way as corpus
        tokenized_query = re.findall(r'\b\w+\b', query.lower())
        
        # Get BM25 scores
        bm25_scores = bm25.get_scores(tokenized_query)
        
        # Pair chunk indices with scores
        keyword_results = [(idx, score) for idx, score in zip(chunk_indices, bm25_scores)]
        
        # Normalize BM25 scores (0-1 range)
        max_score = max(score for _, score in keyword_results) if keyword_results else 1.0
        if max_score > 0:
            keyword_results = [(idx, score / max_score) for idx, score in keyword_results]
    else:
        # Simplified TF-IDF for fallback
        query_tokens = set(re.findall(r'\b\w+\b', query.lower()))
        keyword_results = []
        
        # Calculate token overlap for each chunk
        for i, tokens in enumerate(corpus):
            # Jaccard similarity: intersection / union
            intersection = len(set(tokens) & query_tokens)
            union = len(set(tokens) | query_tokens) if len(set(tokens) | query_tokens) > 0 else 1
            score = intersection / union if union > 0 else 0
            keyword_results.append((chunk_indices[i], score))
    
    # Step 3: Combine both search results with hybrid scoring
    # First, create mapping from result_ids to their vector scores
    vector_scores = {}
    for i, result_id in enumerate(result_ids):
        chunk_idx = int(result_id.split("_")[-1])
        # Convert distance to similarity (1 - distance)
        vector_scores[chunk_idx] = 1.0 - result_distances[i]
    
    # Extract key phrases that should strongly influence ranking
    key_phrases = [
        "grace period", "waiting period", "pre-existing", "maternity", "organ donor",
        "no claim", "health check", "ayush", "room rent", "icu", "hospital", "renewal",
        "continuity", "policy period", "expenses", "covered", "sum insured", "premium", "payment"
    ]
    
    # Identify important phrases in the query
    important_phrases = [phrase for phrase in key_phrases if phrase in query.lower()]
    
    # Final scoring of all candidate chunks
    scored_chunks = []
    seen_page_refs = set()
    seen_chunks_hash = set()
    
    # Create a set of chunks to score (union of vector and keyword results)
    all_candidate_indices = set([int(rid.split("_")[-1]) for rid in result_ids])
    all_candidate_indices.update([idx for idx, _ in sorted(keyword_results, key=lambda x: -x[1])[:vector_k]])
    
    for chunk_idx in all_candidate_indices:
        if chunk_idx >= len(chunks):
            continue
            
        chunk_text = chunks[chunk_idx]
        ref = refs[chunk_idx] if chunk_idx < len(refs) else ""
        
        # Skip if this is an exact duplicate (by hash)
        chunk_hash = hash(chunk_text.strip().lower())
        if chunk_hash in seen_chunks_hash:
            continue
        seen_chunks_hash.add(chunk_hash)
        
        # Get scores from both methods
        vector_score = vector_scores.get(chunk_idx, 0.0)
        keyword_score = 0.0
        for idx, score in keyword_results:
            if idx == chunk_idx:
                keyword_score = score
                break
                
        # Boosting factors
        chunk_lower = chunk_text.lower()
        
        # 1. Direct phrase matching
        phrase_bonus = 0.0
        for phrase in important_phrases:
            if phrase in chunk_lower:
                phrase_bonus += 0.75  # Strong boost for exact phrase match
                
        # 2. Critical insurance term presence
        term_bonus = 0.0
        for term in critical_terms:
            if term in chunk_lower:
                term_bonus += 0.25  # Smaller boost for each term
                
        # 3. Page diversity bonus
        page_bonus = 0.0
        if ref and ref not in seen_page_refs:
            page_bonus = 0.5
            seen_page_refs.add(ref)
        
        # Combined score with balanced weighting
        # Vector: semantic understanding (40%)
        # Keyword: term presence (40%)
        # Bonuses: domain specificity and diversity (20%)
        final_score = (vector_score * 0.4) + (keyword_score * 0.4) + ((phrase_bonus + term_bonus + page_bonus) * 0.2)
        
        scored_chunks.append((final_score, ref, chunk_text))
    
    # Sort by final score (descending)
    scored_chunks.sort(reverse=True)
    
    # Build final result lists with the top chunks
    top_chunks = []
    top_refs = []
    
    for _, ref, chunk_text in scored_chunks:
        top_chunks.append(chunk_text)
        top_refs.append(ref)
        if len(top_chunks) >= top_k:
            break
    return top_chunks, top_refs
