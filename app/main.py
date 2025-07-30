# app/main.py

from dotenv import load_dotenv
load_dotenv()  # Load environment vars from .env file

import os
from fastapi import FastAPI, HTTPException, Security
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import List
from app.document_loader import fetch_and_parse
from app.embedder import get_embeddings
from app.vector_db import upsert_documents, semantic_search
from app.llm_answer import extract_answers
import uuid

app = FastAPI()

# Load API KEY from environment variables
API_KEY = os.getenv("API_KEY")
print("Loaded API_KEY:", API_KEY)  # Confirm loaded key

# Setup HTTP Bearer security scheme
bearer_scheme = HTTPBearer()

# Define request body model using Pydantic
class HackrxRequest(BaseModel):
    documents: str
    questions: List[str]

@app.post("/hackrx/run")
async def hackrx_run(
    payload: HackrxRequest,  # Parsed JSON body will be assigned here
    credentials: HTTPAuthorizationCredentials = Security(bearer_scheme)  # Auth
):
    print("RECEIVED AUTHORIZATION HEADER:", credentials.scheme, credentials.credentials)
    
    # Verify token matches
    if credentials.credentials != API_KEY:
        raise HTTPException(status_code=403, detail="Forbidden: Invalid token")
    
    doc_url = payload.documents
    questions = payload.questions
    
    # Validate request fields (Pydantic does initial validation, this is just extra guard)
    if not doc_url or not questions:
        raise HTTPException(status_code=400, detail="Missing 'documents' or 'questions' in payload")
    
    # Process document (parse, chunk)
    parsed_doc = fetch_and_parse(doc_url)
    doc_chunks = parsed_doc.get("chunks", [])
    
    # Embed chunks
    chunk_texts = [chunk["text"] for chunk in doc_chunks]
    chunk_embeddings = get_embeddings(chunk_texts)
    
    # Create unique Qdrant collection for this request
    collection_id = str(uuid.uuid4())
    upsert_documents(collection_id, doc_chunks, chunk_embeddings)
    
    answers = []
    # For each question: embed, retrieve, extract
    for question in questions:
        question_embedding = get_embeddings([question])[0]
        top_chunks = semantic_search(collection_id, question_embedding, top_k=3)
        
        answer, _ = extract_answers(question, top_chunks)
        answers.append(answer)
    
    # Return answers in JSON format
    return JSONResponse(content={"answers": answers})
