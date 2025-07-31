from dotenv import load_dotenv
load_dotenv()

import os
import hashlib
import time
from fastapi import FastAPI, HTTPException, Security
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import List

from app.document_loader import fetch_and_parse
from app.embedder import get_embeddings
from app.vector_db import collection_exists, upsert_documents, semantic_search
from app.llm_answer import batch_extract_answers_with_retry

app = FastAPI()
API_KEY = os.getenv("API_KEY")
bearer_scheme = HTTPBearer()

class HackrxRequest(BaseModel):
    documents: str
    questions: List[str]

@app.post("/hackrx/run")
async def hackrx_run(
    payload: HackrxRequest,
    credentials: HTTPAuthorizationCredentials = Security(bearer_scheme),
):
    if credentials.credentials != API_KEY:
        raise HTTPException(status_code=403, detail="Forbidden: Invalid token")

    doc_url = payload.documents.strip()
    questions = payload.questions

    if not doc_url or not questions:
        raise HTTPException(status_code=400, detail="Missing 'documents' or 'questions' in payload")

    timings = {}
    t0 = time.time()

    # Create a stable doc id by hashing URL
    doc_id = hashlib.sha256(doc_url.encode()).hexdigest()[:24]

    if not collection_exists(doc_id):
        t_parse = time.time()
        parsed_doc = fetch_and_parse(doc_url)
        doc_chunks = parsed_doc.get("chunks", [])
        if not doc_chunks:
            raise HTTPException(status_code=400, detail="Could not parse document text chunks.")
        timings["parse"] = time.time() - t_parse

        t_embed = time.time()
        chunk_texts = [chunk["text"] for chunk in doc_chunks]
        chunk_embeddings = get_embeddings(chunk_texts)
        timings["embed"] = time.time() - t_embed

        t_upsert = time.time()
        upsert_documents(doc_id, doc_chunks, chunk_embeddings)
        timings["index"] = time.time() - t_upsert

    t_qembed = time.time()
    question_embeddings = get_embeddings(questions)
    timings["question_embed"] = time.time() - t_qembed

    t_search = time.time()
    retrieved_chunks = [semantic_search(doc_id, q_emb, top_k=3) for q_emb in question_embeddings]
    timings["vector_search"] = time.time() - t_search

    # Aggregate unique text chunks to form LLM context
    unique_texts = []
    seen = set()
    for chunks in retrieved_chunks:
        for c in chunks:
            text = c["text"]
            if text not in seen:
                unique_texts.append(text)
                seen.add(text)

    context_for_llm = "\n\n".join(unique_texts[:3])

    t_llm = time.time()
    try:
        answers = await batch_extract_answers_with_retry(questions, context_for_llm)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM call failed: {e}")
    timings["llm"] = time.time() - t_llm

    timings["overall"] = time.time() - t0
    print(f"TIMINGS: {timings}")

    return JSONResponse(content={"answers": answers, "timings": timings})
