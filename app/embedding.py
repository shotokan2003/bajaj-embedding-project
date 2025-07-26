from sentence_transformers import SentenceTransformer
from app.cache import embedding_cache

model = SentenceTransformer("jinaai/jina-embeddings-v2-base-en")  # or any local model

def get_embedding_cached(text: str):
    if text in embedding_cache:
        return embedding_cache[text]
    emb = model.encode(text).tolist()
    embedding_cache[text] = emb
    return emb
