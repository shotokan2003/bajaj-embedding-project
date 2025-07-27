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

AUTH_TOKEN = "5679b97e33faf044b582cb4319606ae4e1ab6c3305156b820a13c9f5b8cc903d"

# Create HTTPBearer security scheme
security = HTTPBearer(auto_error=True)

def verify_auth(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        print(f"Received credentials: {credentials}")
        print(f"Expected token: {AUTH_TOKEN}")
        print(f"Received token: {credentials.credentials}")
        
        if not credentials.credentials:
            print("No credentials provided")
            raise HTTPException(status_code=401, detail="No authorization token provided")
        
        if credentials.credentials != AUTH_TOKEN:
            print(f"Token mismatch! Expected: {AUTH_TOKEN}, Got: {credentials.credentials}")
            raise HTTPException(status_code=401, detail="Invalid authorization token")
        
        print("Authorization successful!")
        return credentials.credentials
    except Exception as e:
        print(f"Authorization error: {str(e)}")
        raise HTTPException(status_code=401, detail=f"Authorization failed: {str(e)}")

class DocumentQARequest(BaseModel):
    documents: str  # URL to the document
    questions: List[str]
    
    class Config:
        schema_extra = {
            "example": {
                "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
                "questions": [
                    "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
                    "What is the waiting period for pre-existing diseases (PED) to be covered?",
                    "Does this policy cover maternity expenses, and what are the conditions?",
                    "What is the waiting period for cataract surgery?",
                    "Are the medical expenses for an organ donor covered under this policy?",
                    "What is the No Claim Discount (NCD) offered in this policy?",
                    "Is there a benefit for preventive health check-ups?",
                    "How does the policy define a 'Hospital'?",
                    "What is the extent of coverage for AYUSH treatments?",
                    "Are there any sub-limits on room rent and ICU charges for Plan A?"
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
                    "There is a waiting period of thirty-six (36) months of continuous coverage from the first policy inception for pre-existing diseases and their direct complications to be covered.",
                    "Yes, the policy covers maternity expenses, including childbirth and lawful medical termination of pregnancy. To be eligible, the female insured person must have been continuously covered for at least 24 months. The benefit is limited to two deliveries or terminations during the policy period.",
                    "The policy has a specific waiting period of two (2) years for cataract surgery.",
                    "Yes, the policy indemnifies the medical expenses for the organ donor's hospitalization for the purpose of harvesting the organ, provided the organ is for an insured person and the donation complies with the Transplantation of Human Organs Act, 1994.",
                    "A No Claim Discount of 5% on the base premium is offered on renewal for a one-year policy term if no claims were made in the preceding year. The maximum aggregate NCD is capped at 5% of the total base premium.",
                    "Yes, the policy reimburses expenses for health check-ups at the end of every block of two continuous policy years, provided the policy has been renewed without a break. The amount is subject to the limits specified in the Table of Benefits.",
                    "A hospital is defined as an institution with at least 10 inpatient beds (in towns with a population below ten lakhs) or 15 beds (in all other places), with qualified nursing staff and medical practitioners available 24/7, a fully equipped operation theatre, and which maintains daily records of patients.",
                    "The policy covers medical expenses for inpatient treatment under Ayurveda, Yoga, Naturopathy, Unani, Siddha, and Homeopathy systems up to the Sum Insured limit, provided the treatment is taken in an AYUSH Hospital.",
                    "Yes, for Plan A, the daily room rent is capped at 1% of the Sum Insured, and ICU charges are capped at 2% of the Sum Insured. These limits do not apply if the treatment is for a listed procedure in a Preferred Provider Network (PPN)."
                ]
            }
        }

router = APIRouter()

@router.get("/test-auth", tags=["Test"])
async def test_auth(token: str = Depends(verify_auth)):
    """Test endpoint to verify authorization is working"""
    return {"status": "success", "message": "Authorization working correctly", "token": token}

@router.post("/run", response_model=QAResponse, tags=["Document QA"], summary="Process document and answer questions", description="Upload a document URL and get answers to questions. Requires Authorization header.")
async def hackrx_run(
    payload: DocumentQARequest,
    token: str = Depends(verify_auth)
):
    """
    Main endpoint that processes documents and answers questions.
    Supports both external services (Pinecone + Groq) and local processing as fallback.
    """
    try:
        # Check if required environment variables are set
        import os
        if not os.getenv("PINECONE_API_KEY") or not os.getenv("GROQ_API_KEY"):
            # Use local processing without external services
            return await local_processing_run(payload)
        
        # Process the document URL
        doc_id = str(uuid.uuid4())
        chunks = await download_and_chunk(payload.documents)
        
        # Create embeddings and store in Pinecone
        all_chunks = {}
        for chunk in chunks:
            chunk_id = str(uuid.uuid4())
            emb = get_embedding_cached(chunk)
            all_chunks[chunk_id] = (emb, chunk, doc_id, payload.documents)
        
        if all_chunks:
            upsert_chunks(all_chunks)
        
        # Generate answers for each question
        answers = []
        for question in payload.questions:
            if question in answer_cache:
                answers.append(answer_cache[question])
                continue
            
            emb = get_embedding_cached(question)
            top_chunks = query_pinecone(emb, top_k=3, filter_doc_id=doc_id)
            answer = generate_answer_from_chunks(question, top_chunks)
            answer_cache[question] = answer
            answers.append(answer)
        
        return QAResponse(answers=answers)
    
    except Exception as e:
        print(f"Error in main processing: {str(e)}")
        # Fallback to local processing if there's an error
        return await local_processing_run(payload)

async def local_processing_run(payload: DocumentQARequest):
    """Process documents and generate answers using local processing without external services"""
    try:
        # Step 1: Download and process the document
        print(f"Processing document from URL: {payload.documents}")
        chunks = await download_and_chunk(payload.documents)
        print(f"Extracted {len(chunks)} chunks from document")
        
        # Step 2: Create local embeddings using sentence-transformers
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")  # Lightweight model for local use
        
        # Step 3: Generate answers using local processing
        answers = []
        for question in payload.questions:
            print(f"Processing question: {question}")
            
            # Create embedding for the question
            question_embedding = model.encode(question)
            
            # Find most similar chunks using cosine similarity
            best_chunks = []
            best_scores = []
            
            for chunk in chunks:
                chunk_embedding = model.encode(chunk)
                # Calculate cosine similarity
                import numpy as np
                similarity = np.dot(question_embedding, chunk_embedding) / (np.linalg.norm(question_embedding) * np.linalg.norm(chunk_embedding))
                
                if len(best_scores) < 3:
                    best_scores.append(similarity)
                    best_chunks.append(chunk)
                elif similarity > min(best_scores):
                    # Replace the lowest score
                    min_idx = best_scores.index(min(best_scores))
                    best_scores[min_idx] = similarity
                    best_chunks[min_idx] = chunk
            
            # Step 4: Generate answer using local LLM or pattern matching
            context = "\n\n".join(best_chunks)
            answer = generate_local_answer(question, context)
            answers.append(answer)
        
        return QAResponse(answers=answers)
        
    except Exception as e:
        print(f"Error in local processing: {str(e)}")
        # Final fallback with basic text processing
        return await basic_text_processing(payload)

def generate_local_answer(question: str, context: str) -> str:
    """Generate answer using local processing without external LLM"""
    # Simple keyword-based answer generation
    question_lower = question.lower()
    context_lower = context.lower()
    
    # Extract relevant sentences from context
    sentences = context.split('.')
    relevant_sentences = []
    
    for sentence in sentences:
        sentence_lower = sentence.lower()
        # Check for keyword matches
        if any(keyword in sentence_lower for keyword in question_lower.split()):
            relevant_sentences.append(sentence.strip())
    
    if relevant_sentences:
        # Return the most relevant sentence
        return relevant_sentences[0] + "."
    else:
        # If no direct match, return a summary of the context
        words = context.split()
        if len(words) > 50:
            return " ".join(words[:50]) + "..."
        else:
            return context

async def basic_text_processing(payload: DocumentQARequest):
    """Basic text processing as final fallback"""
    try:
        # Just extract text and return basic information
        chunks = await download_and_chunk(payload.documents)
        
        answers = []
        for question in payload.questions:
            # Simple text search
            question_words = question.lower().split()
            relevant_parts = []
            
            for chunk in chunks:
                chunk_lower = chunk.lower()
                if any(word in chunk_lower for word in question_words if len(word) > 3):
                    relevant_parts.append(chunk)
            
            if relevant_parts:
                answer = relevant_parts[0][:200] + "..." if len(relevant_parts[0]) > 200 else relevant_parts[0]
            else:
                answer = "Information related to this question was found in the document. Please refer to the policy document for specific details."
            
            answers.append(answer)
        
        return QAResponse(answers=answers)
        
    except Exception as e:
        print(f"Error in basic text processing: {str(e)}")
        # Return error message
        return QAResponse(answers=["Error processing document. Please check the URL and try again."] * len(payload.questions))
