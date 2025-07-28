from sentence_transformers import SentenceTransformer
from app.cache import embedding_cache, get_embedding_from_redis, set_embedding_to_redis
import numpy as np
import asyncio

# model = SentenceTransformer("jinaai/jina-embeddings-v2-base-en")  # or any local model
model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

def get_embedding_cached(text: str):
    if text in embedding_cache:
        return embedding_cache[text]
    # Try Redis cache (async)
    loop = asyncio.get_event_loop()
    redis_emb = loop.run_until_complete(get_embedding_from_redis(text))
    if redis_emb:
        emb = np.fromstring(redis_emb, sep=',').tolist()
        embedding_cache[text] = emb
        return emb
    emb = model.encode(text).tolist()
    embedding_cache[text] = emb
    # Set in Redis (async)
    loop.run_until_complete(set_embedding_to_redis(text, emb))
    return emb
