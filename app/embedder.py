from fastembed import TextEmbedding

# Initialize once with chosen model
_embedder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5", batch_size=256, parallel=0)

def get_embeddings(text_list):
    """
    Embed a list of texts and return list of vectors.
    """
    # embed() returns a generator, so convert to list
    return list(_embedder.embed(text_list))
