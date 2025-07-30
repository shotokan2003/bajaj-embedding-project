# app/vector_db.py

from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance

# Connect to Qdrant vector DB (adjust host/port as needed)
_qdrant_client = QdrantClient(host="localhost", port=6333)

def upsert_documents(collection_name, doc_chunks, embeddings):
    """
    Creates or recreates a collection and inserts document chunk embeddings with metadata.
    
    :param collection_name: str unique collection id
    :param doc_chunks: List[Dict] each chunk with 'text' and optionally 'section_id'
    :param embeddings: List[List[float]] embedding vectors corresponding to chunks
    """
    vector_size = len(embeddings[0])  # Dimension of embedding vectors
    
    # Delete collection if exists to start fresh
    if _qdrant_client.collection_exists(collection_name):
        _qdrant_client.delete_collection(collection_name)
    
    # Recreate collection with vector config (size and distance metric)
    _qdrant_client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )
    
    # Prepare points to upsert with payload metadata
    points = []
    for idx, e in enumerate(embeddings):
        points.append({
            "id": idx,
            "vector": e,
            "payload": {
                "text": doc_chunks[idx]["text"],
                "section_id": doc_chunks[idx].get("section_id")
            }
        })
    
    # Upsert vectors + metadata into collection
    _qdrant_client.upsert(collection_name=collection_name, points=points)

def semantic_search(collection_name, query_vector, top_k=3):
    """
    Perform semantic vector search on stored chunks.
    
    :param collection_name: str Qdrant collection name
    :param query_vector: List[float] embedding vector of the query
    :param top_k: int number of results to retrieve
    :return: List of dicts with keys 'text', 'section_id', 'similarity'
    """
    results = _qdrant_client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=top_k
    )
    
    return [
        {
            "text": res.payload["text"],
            "section_id": res.payload.get("section_id"),
            "similarity": res.score
        }
        for res in results
    ]
