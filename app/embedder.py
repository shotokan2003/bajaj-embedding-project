from fastembed import TextEmbedding

# Initialize FastEmbed model once
_embedder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5", batch_size=256, parallel=0)

def get_embeddings(text_list):
    """
    Generate embeddings for a list of texts.
    Returns a list of vectors (lists of floats).
    """
    return list(_embedder.embed(text_list))
