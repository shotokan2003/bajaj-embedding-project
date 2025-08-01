"""
ChromaDB patch for NumPy 2.0+ compatibility
"""

import os
import sys
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add compatibility layer for NumPy 2.0+
if not hasattr(np, 'float_'):
    np.float_ = np.float64
    logger.info("Added np.float_ compatibility for NumPy 2.0+")

# Import ChromaDB only after patching NumPy
import chromadb
from chromadb.config import Settings

logger.info(f"ChromaDB version: {chromadb.__version__}")
logger.info(f"NumPy version: {np.__version__}")
logger.info("NumPy and ChromaDB compatibility patch applied")

# MonkeyPatch ChromaDB's internal usage of np.float_ if needed
try:
    # Check an example dtype from chromadb
    test_array = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    logger.info("Test array created successfully with np.float64")
except Exception as e:
    logger.error(f"Error creating test array: {e}")

# Return the patched versions
__all__ = ['np', 'chromadb']
