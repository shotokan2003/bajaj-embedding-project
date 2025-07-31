from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance

_qdrant_client = QdrantClient(host="localhost", port=6333)

def collection_exists(collection_name):
    return _qdrant_client.collection_exists(collection_name)

def upsert_documents(collection_name, doc_chunks, embeddings):
    vector_size = len(embeddings)
    if _qdrant_client.collection_exists(collection_name):
        _qdrant_client.delete_collection(collection_name)
    _qdrant_client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
    )
    points = [
        {
            "id": idx,
            "vector": e,
            "payload": {
                "text": doc_chunks[idx]["text"],
                "section_id": doc_chunks[idx].get("section_id")
            }
        }
        for idx, e in enumerate(embeddings)
    ]
    _qdrant_client.upsert(collection_name=collection_name, points=points)

def semantic_search(collection_name, query_vector, top_k=3):
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
