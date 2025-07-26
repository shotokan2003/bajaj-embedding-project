from cachetools import LRUCache

embedding_cache = LRUCache(maxsize=2000)
answer_cache = LRUCache(maxsize=1000)
