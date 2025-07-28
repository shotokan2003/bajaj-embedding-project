from cachetools import LRUCache

embedding_cache = LRUCache(maxsize=2000)
answer_cache = LRUCache(maxsize=1000)

import asyncio
from coredis import Redis

# Async Redis client (singleton pattern for reuse)
redis_client: Redis = None

async def get_redis_client():
    global redis_client
    if redis_client is None:
        redis_client = Redis(host='localhost', port=6379, db=0, encoding='utf-8', decode_responses=True)
    return redis_client

# Async Redis cache helpers
async def get_embedding_from_redis(text: str):
    client = await get_redis_client()
    return await client.get(f'embedding:{text}')

async def set_embedding_to_redis(text: str, embedding):
    client = await get_redis_client()
    # Store as string (e.g., comma-separated floats)
    await client.set(f'embedding:{text}', ','.join(map(str, embedding)))

async def get_answer_from_redis(question: str):
    client = await get_redis_client()
    return await client.get(f'answer:{question}')

async def set_answer_to_redis(question: str, answer: str):
    client = await get_redis_client()
    await client.set(f'answer:{question}', answer)
