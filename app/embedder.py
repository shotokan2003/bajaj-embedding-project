# from fastembed import TextEmbedding

# # Initialize FastEmbed model once
# _embedder = TextEmbedding(model_name="BAAI/bge-large-en-v1.5", batch_size=256, parallel=2)

# def get_embeddings(text_list):
#     """
#     Generate embeddings for a list of texts.
#     Returns a list of vectors (lists of floats).
#     """
#     return list(_embedder.embed(text_list))






from google import generativeai as genai
import os

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
EMBED_MODEL = "gemini-embedding-001"

def get_embeddings(text_list):
    embeddings = []
    for text in text_list:
        result = genai.embed_content(model=EMBED_MODEL, content=text)
        embeddings.append(result["embedding"])
    return embeddings






