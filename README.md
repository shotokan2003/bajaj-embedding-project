# HackRx LLM Query API

A high-performance, modular FastAPI backend for LLM-powered document Q&A with vector search, chunking, and caching.

## Structure
- `app/main.py`: FastAPI app and endpoint
- `app/pipeline.py`: Orchestrates the pipeline
- `app/vector_store.py`: Vector search with Google Gemini embeddings (ChromaDB)
- `app/cloud_embeddings.py`: Google Generative AI (Gemini) embedding API integration
- `app/llm.py`: LLM calls (OpenAI/Ollama)
- `app/cache.py`: Redis/in-memory cache
- `app/utils.py`: Download, parse, chunk docs

## Quickstart
1. Install requirements:
   ```powershell
   pip install -r requirements.txt
   ```
2. Run the server:
   ```powershell
   uvicorn app.main:app --reload
   ```
3. POST to `/hackrx/run` with Bearer token and JSON body as described in the docs.

## .env
- `HACKRX_API_TOKEN` (default: supersecrettoken)
- `OPENAI_API_KEY` (for GPT-4o)
- `GOOGLE_API_KEY` (for embedding API)
- `GROQ_API_KEY` (for fast LLM responses)
- `USE_OLLAMA` (set to 1 for local Llama3)
- `USE_REDIS` (set to 1 to enable Redis caching)

## Vercel Deployment
This project is optimized for serverless deployment on Vercel:

1. Remove heavy ML dependencies by using Google's embedding API instead of local models
2. Optimize package size to meet Vercel's 50MB limit
3. Structure API endpoints according to Vercel's serverless function pattern

## Testing Gemini Embeddings
To verify the Google Gemini embedding API integration:
```powershell
python test_cloud_embeddings.py
```

---
