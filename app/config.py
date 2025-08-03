"""
Configuration module for optimized chunking and embedding parameters.
This allows easy tuning of the system for different document types and use cases.
"""

import os
from typing import Dict, Any

class OptimizationConfig:
    """Configuration class for chunking and embedding optimization"""
    
    # Chunking parameters
    CHUNK_SIZE = 500  # Reduced further for speed (was 600)
    OVERLAP_SIZE = 75   # Reduced overlap for speed (was 100)
    MIN_CHUNK_SIZE = 50  # Minimum chunk size in words
    MAX_CHUNK_SIZE = 800  # Reduced max size (was 1000)
    MAX_CHUNKS = 300  # Reduced max chunks for faster processing (was 500)
    
    # Embedding parameters
    EMBEDDING_BATCH_SIZE = 16  # Reduced for faster processing
    MAX_CONCURRENT_EMBEDDINGS = 16  # Reduced for better rate limiting
    EMBEDDING_RETRY_ATTEMPTS = 2  # Reduced retry attempts
    EMBEDDING_CHUNK_SIZE = 6  # Smaller chunks for faster API processing
    
    # Retrieval parameters
    DEFAULT_TOP_K = 8   # Reduced from 12 for speed
    RERANK_TOP_K = 5    # Reduced from 6 for speed
    SEMANTIC_WEIGHT = 0.65  # Increased semantic weight for faster processing
    KEYWORD_WEIGHT = 0.20   # Reduced keyword weight
    POSITION_WEIGHT = 0.10  # Same
    TYPE_WEIGHT = 0.05      # Same
    
    # Performance parameters
    MAX_WORKERS = min(32, os.cpu_count() + 4)
    REQUEST_TIMEOUT = 30
    RATE_LIMIT_DELAY = 0.1
    
    # Document processing parameters
    PARALLEL_PAGES = True
    EXTRACT_TABLES = True
    PRESERVE_FORMATTING = True
    
    # Critical sections for insurance documents
    INSURANCE_CRITICAL_PATTERNS = {
        "grace_period": [
            r'(?i)grace period[^.]*?(?:days?|months?|period)[^.]*?\.',
            r'(?i)premium.*?due.*?(?:days?|months?)[^.]*?\.',
            r'(?i)renewal.*?grace.*?(?:days?|months?)[^.]*?\.'
        ],
        "waiting_period": [
            r'(?i)waiting period[^.]*?(?:months?|years?|days?)[^.]*?\.',
            r'(?i)pre-existing.*?(?:months?|years?|waiting)[^.]*?\.',
            r'(?i)diseases?.*?waiting.*?(?:months?|years?)[^.]*?\.'
        ],
        "maternity": [
            r'(?i)maternity[^.]*?(?:covered?|coverage|expenses?|benefits?)[^.]*?\.',
            r'(?i)pregnancy[^.]*?(?:covered?|coverage|waiting)[^.]*?\.',
            r'(?i)childbirth[^.]*?(?:covered?|coverage|expenses?)[^.]*?\.'
        ],
        "ayush": [
            r'(?i)(?:ayush|ayurveda|yoga|naturopathy|unani|siddha|homeopathy)[^.]*?(?:covered?|coverage|treatment)[^.]*?\.',
            r'(?i)alternative.*?medicine[^.]*?(?:covered?|coverage)[^.]*?\.'
        ],
        "room_rent": [
            r'(?i)room.*?(?:rent|charges?)[^.]*?(?:covered?|limit|rupees?)[^.]*?\.',
            r'(?i)icu.*?charges?[^.]*?(?:covered?|limit|rupees?)[^.]*?\.'
        ],
        "exclusions": [
            r'(?i)(?:not covered?|excluded?|exclusions?)[^.]*?\.',
            r'(?i)policy.*?(?:does not cover|excludes)[^.]*?\.'
        ]
    }
    
    # Question type patterns for better retrieval
    QUESTION_PATTERNS = {
        'coverage': ['covered', 'cover', 'coverage', 'include', 'eligible'],
        'waiting': ['waiting', 'wait', 'period', 'duration', 'months', 'years'],
        'grace': ['grace', 'period', 'due', 'payment', 'premium'],
        'amount': ['amount', 'limit', 'rupees', 'cost', 'price', 'charges'],
        'exclusion': ['not covered', 'excluded', 'exclusion', 'does not cover'],
        'procedure': ['procedure', 'treatment', 'surgery', 'operation', 'therapy']
    }
    
    @classmethod
    def get_config(cls) -> Dict[str, Any]:
        """Get all configuration as a dictionary"""
        return {
            'chunking': {
                'chunk_size': cls.CHUNK_SIZE,
                'overlap_size': cls.OVERLAP_SIZE,
                'min_chunk_size': cls.MIN_CHUNK_SIZE,
                'max_chunk_size': cls.MAX_CHUNK_SIZE
            },
            'embedding': {
                'batch_size': cls.EMBEDDING_BATCH_SIZE,
                'max_concurrent': cls.MAX_CONCURRENT_EMBEDDINGS,
                'retry_attempts': cls.EMBEDDING_RETRY_ATTEMPTS
            },
            'retrieval': {
                'default_top_k': cls.DEFAULT_TOP_K,
                'rerank_top_k': cls.RERANK_TOP_K,
                'semantic_weight': cls.SEMANTIC_WEIGHT,
                'keyword_weight': cls.KEYWORD_WEIGHT,
                'position_weight': cls.POSITION_WEIGHT,
                'type_weight': cls.TYPE_WEIGHT
            },
            'performance': {
                'max_workers': cls.MAX_WORKERS,
                'request_timeout': cls.REQUEST_TIMEOUT,
                'rate_limit_delay': cls.RATE_LIMIT_DELAY
            }
        }
    
    @classmethod
    def update_config(cls, **kwargs):
        """Update configuration parameters"""
        for key, value in kwargs.items():
            if hasattr(cls, key.upper()):
                setattr(cls, key.upper(), value)

# Environment-based configuration
environment = os.getenv("ENVIRONMENT", "speed")  # Default to speed optimization

if environment == "production":
    # Production optimizations
    OptimizationConfig.EMBEDDING_BATCH_SIZE = 16
    OptimizationConfig.MAX_CONCURRENT_EMBEDDINGS = 16
    OptimizationConfig.RATE_LIMIT_DELAY = 0.2
elif environment == "development":
    # Development optimizations
    OptimizationConfig.EMBEDDING_BATCH_SIZE = 8
    OptimizationConfig.MAX_CONCURRENT_EMBEDDINGS = 8
    OptimizationConfig.RATE_LIMIT_DELAY = 0.05
elif environment == "speed":
    # Ultra-speed optimizations for <25s target
    OptimizationConfig.CHUNK_SIZE = 1000             # Larger chunks = fewer total chunks
    OptimizationConfig.CHUNK_OVERLAP = 50            # Minimal overlap  
    OptimizationConfig.MAX_CHUNKS = 60               # Drastically reduced
    OptimizationConfig.EMBEDDING_BATCH_SIZE = 8      # Smaller batches for faster processing
    OptimizationConfig.DEFAULT_TOP_K = 6             # Fewer candidates
    OptimizationConfig.RERANK_TOP_K = 3              # Minimal final chunks
    OptimizationConfig.RATE_LIMIT_DELAY = 0.01       # Minimal delays
    OptimizationConfig.MAX_CONCURRENT_EMBEDDINGS = 12 # Higher concurrency

# Export the config instance
config = OptimizationConfig()
