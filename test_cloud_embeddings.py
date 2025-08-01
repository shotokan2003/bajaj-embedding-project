# Apply NumPy patch first - this must be at the very top
import sys
import os

# Add app directory to path to find the numpy_patch module
sys.path.insert(0, os.path.dirname(__file__))

# Import our custom NumPy patch for compatibility with ChromaDB
from app.numpy_patch import np

import os
import sys
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add app directory to path
sys.path.append(os.path.join(os.path.dirname(__file__)))

# Load environment variables
load_dotenv()

def test_cloud_embeddings():
    """Test that Gemini embeddings are working correctly."""
    try:
        from app.cloud_embeddings import encode
        
        # Test single embedding
        test_text = "This is a test sentence to check if Google Gemini embeddings are working correctly."
        logger.info("Testing single text embedding...")
        embedding = encode([test_text])
        
        # Check shape and values
        logger.info(f"Embedding shape: {embedding.shape}")
        logger.info(f"First few values: {embedding[0][:5]}")
        
        # Test batch embedding
        test_texts = [
            "This is the first test sentence.",
            "Here is another test sentence to embed.",
            "Let's check if batching works correctly."
        ]
        
        logger.info("Testing batch embedding...")
        batch_embeddings = encode(test_texts)
        logger.info(f"Batch embeddings shape: {batch_embeddings.shape}")
          # Check if embeddings are different (they should be)
        cos_sim = np.dot(batch_embeddings[0], batch_embeddings[1]) / (np.linalg.norm(batch_embeddings[0]) * np.linalg.norm(batch_embeddings[1]))
        logger.info(f"Cosine similarity between embeddings 1 and 2: {cos_sim}")
        
        logger.info("Gemini embeddings test completed successfully!")
        return True
    except Exception as e:
        logger.error(f"Error testing Gemini embeddings: {str(e)}")
        return False

def test_vector_store():
    """Test that vector store works with Google Vertex AI embeddings."""
    try:
        from app.vector_store import get_or_create_embeddings, retrieve_similar_chunks
        
        # Create test document
        test_doc_url = "https://test-document.com/doc1"
        test_chunks = [
            "This is a document about insurance policies.",
            "The grace period is 30 days for premium payment.",
            "Pre-existing diseases have a waiting period of 36 months.",
            "Health checkups are provided after 2 continuous policy years."
        ]
        test_refs = ["p1", "p2", "p3", "p4"]
        
        logger.info("Testing document embedding storage...")
        doc_id, embeddings = get_or_create_embeddings(test_doc_url, test_chunks, test_refs)
        
        logger.info(f"Document ID: {doc_id}")
        logger.info(f"Embeddings shape: {embeddings.shape}")
        
        # Test retrieval
        test_query = "What is the grace period for premium payment?"
        logger.info(f"Testing retrieval for query: {test_query}")
        
        similar_chunks, similar_refs = retrieve_similar_chunks(
            doc_id, test_query, test_chunks, test_refs, embeddings
        )
        
        logger.info("Retrieved chunks:")
        for i, (chunk, ref) in enumerate(zip(similar_chunks, similar_refs)):
            logger.info(f"{i+1}. [{ref}] {chunk}")
        
        logger.info("Vector store test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error testing vector store: {str(e)}")
        return False

if __name__ == "__main__":
    logger.info("=== Testing Cloud Embeddings Integration ===")
    
    # Apply NumPy 2.0+ compatibility fixes
    if not hasattr(np, 'float_'):
        np.float_ = np.float64
        logger.info("Applied NumPy 2.0+ compatibility patch (np.float_ = np.float64)")
    
    try:
        cloud_test_success = test_cloud_embeddings()
        logger.info("Cloud embeddings test completed successfully!")
        
        if cloud_test_success:
            try:
                vector_store_success = test_vector_store()
                logger.info("Vector store test completed successfully!")
            except Exception as e:
                import traceback
                logger.error(f"Vector store test failed with error: {str(e)}")
                logger.error(traceback.format_exc())
                
                # Try to recover from NumPy errors
                if "float_" in str(e):
                    logger.info("Detected NumPy float_ issue - attempting recovery...")
                    # Try applying additional NumPy 2.0+ compatibility fixes
                    import sys
                    sys.modules['numpy'].float_ = sys.modules['numpy'].float64
                    try:
                        logger.info("Re-running vector store test after fix...")
                        vector_store_success = test_vector_store()
                        logger.info("Vector store test completed successfully after fix!")
                    except Exception as e2:
                        logger.error(f"Recovery failed: {str(e2)}")
                        vector_store_success = False
                else:
                    vector_store_success = False
        else:
            vector_store_success = False
            
        if cloud_test_success and vector_store_success:
            logger.info("✅ All tests passed! Cloud embeddings integration is working correctly.")
        else:
            logger.error("❌ Tests failed. Please check the error logs.")
    except Exception as e:
        import traceback
        logger.error(f"Test execution failed with error: {str(e)}")
        logger.error(traceback.format_exc())
