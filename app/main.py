"""
Main FastAPI app for /hackrx/run endpoint.
Handles authentication, request validation, and response formatting.
"""

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from app.pipeline import process_document_and_answer
from contextlib import asynccontextmanager
import os
import logging
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Define lifespan for startup/shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize background tasks
    from app.background_tasks import start_background_tasks, stop_background_tasks
    start_background_tasks()
    
    # Clear stale cache entries on startup
    from app.cache import clear_stale_cache
    removed = clear_stale_cache(max_age_days=7)
    logging.info(f"Cleared {removed} stale cache entries")
    yield
    
    # Stop background tasks on shutdown
    stop_background_tasks()

app = FastAPI(title="HackRx LLM Query API", lifespan=lifespan)
# security = HTTPBearer()

# Dummy token for hackathon; replace with env/config in prod
# API_TOKEN = os.getenv("HACKRX_API_TOKEN", "supersecrettoken")

class QueryRequest(BaseModel):
    documents: str
    questions: list[str]

class QueryResponse(BaseModel):
    answers: list[str]

# def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
#     if credentials.credentials != API_TOKEN:
#         raise HTTPException(status_code=401, detail="Invalid or missing token.")

@app.post("/hackrx/run", response_model=QueryResponse)
async def hackrx_run(
    req: QueryRequest,
    # credentials: HTTPAuthorizationCredentials = Depends(verify_token)
):
    """
    Main endpoint: Accepts a document URL and questions, returns answers.
    """
    start_time = time.time()
    logging.info(f"Received request with {len(req.questions)} questions")
    
    try:
        answers = await process_document_and_answer(req.documents, req.questions)
        elapsed = time.time() - start_time
        logging.info(f"Request processed in {elapsed:.2f} seconds")
        return {"answers": answers}
    except Exception as e:
        logging.error(f"Error processing request: {str(e)}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # Run the FastAPI app with Uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), reload=True)
