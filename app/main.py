"""
Main FastAPI app for /hackrx/run endpoint.
Handles authentication, request validation, and response formatting.
"""

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from app.pipeline import process_document_and_answer
import os

app = FastAPI(title="HackRx LLM Query API")
security = HTTPBearer()

# Dummy token for hackathon; replace with env/config in prod
API_TOKEN = os.getenv("HACKRX_API_TOKEN", "supersecrettoken")

class QueryRequest(BaseModel):
    documents: str
    questions: list[str]

class QueryResponse(BaseModel):
    answers: list[str]

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != API_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid or missing token.")

@app.post("/hackrx/run", response_model=QueryResponse)
async def hackrx_run(
    req: QueryRequest,
    credentials: HTTPAuthorizationCredentials = Depends(verify_token)
):
    """
    Main endpoint: Accepts a document URL and questions, returns answers.
    """
    try:
        answers = await process_document_and_answer(req.documents, req.questions)
        return {"answers": answers}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
