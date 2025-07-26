import app.config  # Ensures .env is loaded
import os
from pinecone import Pinecone

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV", "us-west1-gcp")
INDEX_NAME = os.getenv("PINECONE_INDEX", "hackrx-index")

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

def query_pinecone(vector: list, top_k: int = 3, filter_doc_id: str = None):
    query_kwargs = {
        "vector": vector,
        "top_k": top_k,
        "include_metadata": True
    }
    if filter_doc_id:
        query_kwargs["filter"] = {"document_id": {"$eq": filter_doc_id}}
    query_response = index.query(**query_kwargs)
    matches = [match['metadata']['text'] for match in query_response['matches']]
    return matches

def upsert_chunks(id_to_embedding_and_text: dict):
    """
    id_to_embedding_and_text: Dict with chunk_id: (embedding, text, document_id, filename)
    """
    vectors = [
        (k, v[0], {"text": v[1], "document_id": v[2], "filename": v[3]})
        for k, v in id_to_embedding_and_text.items()
    ]
    index.upsert(vectors)
