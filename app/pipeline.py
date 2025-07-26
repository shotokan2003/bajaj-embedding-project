"""
Pipeline module: Orchestrates document ingestion, chunking, embedding, retrieval, LLM, and caching.
Optimized for parallel processing and speed.
"""

from app.utils import download_and_parse_document, chunk_text
from app.vector_store import get_or_create_embeddings, retrieve_similar_chunks_async
from app.llm import answer_with_llm
from app.cache import get_cached_answer, cache_answer
import asyncio
import time
from typing import List, Dict, Any, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def process_document_and_answer(doc_url: str, questions: list[str]) -> list[str]:
    """
    Main pipeline: For a document and list of questions, returns answers.
    Optimized for parallel processing of document parsing and question answering.
    """
    start_time = time.time()
    logger.info(f"Processing document: {doc_url}")
    logger.info(f"Number of questions: {len(questions)}")
    
    # Step 1: Download and parse document (PDF/DOCX) - async with caching
    text, meta = await download_and_parse_document(doc_url)
    if not text:
        raise ValueError("Document parsing failed or empty document.")
    logger.info(f"Document parsed in {time.time() - start_time:.2f}s")
    
    # Step 2: Chunk text - optimized for semantic boundaries
    chunks, chunk_refs = chunk_text(text, meta)
    logger.info(f"Document chunked into {len(chunks)} segments")
    
    # Step 3: Get or create embeddings and store in ChromaDB
    embedding_start = time.time()
    doc_id, embeddings = get_or_create_embeddings(doc_url, chunks)
    logger.info(f"Embeddings processed in {time.time() - embedding_start:.2f}s")
    
    # Step 4: For each question, process in parallel
    query_start = time.time()
    
    # First, check cache for all questions
    cached_answers = {}
    questions_to_process = []
    
    for i, q in enumerate(questions):
        cached = get_cached_answer(doc_url, q)
        if cached:
            cached_answers[i] = cached
        else:
            questions_to_process.append((i, q))
    
    logger.info(f"Cache hits: {len(cached_answers)}/{len(questions)}")
    
    # Process remaining questions in parallel
    if questions_to_process:
        # Step 1: Retrieve contexts for all questions in parallel
        retrieval_tasks = []
        for i, q in questions_to_process:
            task = retrieve_similar_chunks_async(doc_id, q, chunks, chunk_refs, top_k=3)
            retrieval_tasks.append((i, q, task))
        
        # Wait for all retrieval tasks
        context_results = {}
        for i, q, task in retrieval_tasks:
            top_chunks, top_refs = await task
            context_results[i] = (q, top_chunks, top_refs)
        
        # Step 2: Generate answers with LLM in parallel
        llm_tasks = []
        for i, (q, top_chunks, top_refs) in context_results.items():
            task = asyncio.create_task(answer_with_llm(q, top_chunks, top_refs))
            llm_tasks.append((i, q, task))
        
        # Wait for all LLM tasks
        for i, q, task in llm_tasks:
            answer = await task
            cached_answers[i] = answer
            # Cache the new answer
            cache_answer(doc_url, q, answer)
    
    # Prepare final answers in the original order
    answers = [cached_answers.get(i) for i in range(len(questions))]
    
    logger.info(f"Questions answered in {time.time() - query_start:.2f}s")
    logger.info(f"Total processing time: {time.time() - start_time:.2f}s")
    
    return answers
