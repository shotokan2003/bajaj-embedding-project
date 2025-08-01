"""
NumPy 2.0+ compatibility patch

This patch adds backward compatibility for libraries that still use 
deprecated NumPy attributes like np.float_ which was removed in NumPy 2.0
"""

import sys
import logging
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add np.float_ if it doesn't exist (NumPy 2.0+)
if not hasattr(np, 'float_'):
    # Add float_ as an alias to float64
    np.float_ = np.float64
    logger.info("Added np.float_ compatibility for NumPy 2.0+")

    # Patch the module dictionary to ensure imports from numpy also work
    sys.modules['numpy'].float_ = np.float64
    logger.info("Patched sys.modules['numpy'] for float_ compatibility")

# Return the patched numpy module
__all__ = ['np']
