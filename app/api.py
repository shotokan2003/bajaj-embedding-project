from fastapi import APIRouter, Header, HTTPException, UploadFile, File, Form, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import List, Optional, Union
from app.document_processing import download_and_chunk, extract_text_from_pdf, chunk_text
from app.embedding import get_embedding_cached
from app.pinecone_client import query_pinecone, upsert_chunks
from app.answer_generation import generate_answer_from_chunks
from app.cache import answer_cache
import uuid

AUTH_TOKEN = "mysecretkey11"

class QARequest(BaseModel):
    document_id: str
    questions: List[str]
    class Config:
        schema_extra = {
            "example": {
                "document_id": "uuid-of-uploaded-document",
                "questions": [
                    "What is the grace period for premium payment?",
                    "What is the waiting period for pre-existing diseases?"
                ]
            }
        }

class QAResponse(BaseModel):
    answers: List[str]
    class Config:
        schema_extra = {
            "example": {
                "answers": [
                    "A grace period of thirty days is provided for premium payment after the due date to renew or continue the policy without losing continuity benefits.",
                    "There is a waiting period of thirty-six (36) months of continuous coverage from the first policy inception for pre-existing diseases and their direct complications to be covered."
                ]
            }
        }

class UploadResponse(BaseModel):
    message: str
    processed_docs: int
    document_ids: list[str]
    filenames: list[str]

router = APIRouter()

security = HTTPBearer()

@router.post("/run", response_model=QAResponse, tags=["QA"], summary="Ask questions about uploaded documents", description="Provide a document_id and a list of questions. Requires Authorization header.")
async def hackrx_run(
    payload: QARequest,
    authorization: Optional[str] = Header(None)
):
    if authorization != f"Bearer {AUTH_TOKEN}":
        raise HTTPException(status_code=401, detail="Unauthorized")
    # Query Pinecone for chunks with this document_id
    answers = []
    for question in payload.questions:
        if question in answer_cache:
            answers.append(answer_cache[question])
            continue
        emb = get_embedding_cached(question)
        top_chunks = query_pinecone(emb, top_k=3, filter_doc_id=payload.document_id)
        print(f"Top chunks for question '{question}': {top_chunks}")
        answer = generate_answer_from_chunks(question, top_chunks)
        answer_cache[question] = answer
        answers.append(answer)
    return QAResponse(answers=answers)

# Alias route for /hackrx/run
from fastapi import APIRouter as FastAPIRouter
alt_router = FastAPIRouter()

@alt_router.post("/hackrx/run", response_model=QAResponse, tags=["QA"], summary="Ask questions about uploaded documents (alias)", description="Same as /api/v1/hackrx/run. Provide a document_id and a list of questions. Requires Authorization header.")
async def hackrx_run_alias(
    payload: QARequest,
    authorization: Optional[str] = Header(None)
):
    return await hackrx_run(payload, authorization)

@router.post("/upload", response_model=UploadResponse)
async def upload_documents(
    urls: Union[str, None] = Form(None),
    files: Union[list[UploadFile], None] = File(None),
    authorization: Optional[str] = Header(None)
):
    if authorization != f"Bearer {AUTH_TOKEN}":
        raise HTTPException(status_code=401, detail="Unauthorized")

    all_chunks = {}
    processed_docs = 0
    document_ids = []
    filenames = []

    # Process URLs
    if urls:
        url_list = [u.strip() for u in urls.split(",") if u.strip()]
        for url in url_list:
            doc_id = str(uuid.uuid4())
            chunks = await download_and_chunk(url)
            for chunk in chunks:
                chunk_id = str(uuid.uuid4())
                emb = get_embedding_cached(chunk)
                all_chunks[chunk_id] = (emb, chunk, doc_id, url)
            document_ids.append(doc_id)
            filenames.append(url)
            processed_docs += 1

    # Process uploaded files
    if files:
        import io
        for file in files:
            content = await file.read()
            pdf_file = io.BytesIO(content)
            text = extract_text_from_pdf(pdf_file)
            doc_id = str(uuid.uuid4())
            for chunk in chunk_text(text):
                chunk_id = str(uuid.uuid4())
                emb = get_embedding_cached(chunk)
                all_chunks[chunk_id] = (emb, chunk, doc_id, file.filename)
            document_ids.append(doc_id)
            filenames.append(file.filename)
            processed_docs += 1

    if all_chunks:
        upsert_chunks(all_chunks)

    return UploadResponse(message="Documents processed and embeddings stored.", processed_docs=processed_docs, document_ids=document_ids, filenames=filenames)
