# app/embedder.py

from fastembed import TextEmbedding

# Initialize the FastEmbed model once.
# You can specify different models if needed; here we use "BAAI/bge-small-en-v1.5" which is lightweight and fast.
_embedder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5", batch_size=256, parallel=0)

def get_embeddings(text_list):
    """
    Generate embeddings for a batch of input texts.

    Args:
        text_list (List[str]): List of textual documents or queries to embed.

    Returns:
        List[List[float]]: List of embedding vectors corresponding to each input text.
    """
    # The embed method returns a generator of embeddings. Convert it to a list for easier use.
    return list(_embedder.embed(text_list))
