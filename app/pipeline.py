"""
Pipeline module: Orchestrates document ingestion, chunking, embedding, retrieval, and LLM.
Optimized for parallel processing and speed with direct processing (no caching).
"""

from app.utils import download_and_parse_document, chunk_text
from app.vector_store import get_or_create_embeddings, retrieve_similar_chunks_async, DEFAULT_TOP_K
from app.llm import answer_with_llm, optimize_prompt
from app.llm import USE_OLLAMA
from app.config import config
import asyncio
import time
from typing import List, Dict, Any, Tuple
import logging
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def post_process_answer(answer: str) -> str:
    """
    Enhanced answer post-processing for better accuracy and completeness.
    Ensures answers are properly formatted while preserving important details.
    """
    if not answer or not answer.strip():
        return "Information not available in the provided document."
    
    # Remove reference phrasing but preserve the actual answer
    answer = re.sub(r'^According to .*?, ', '', answer, flags=re.IGNORECASE)
    answer = re.sub(r'^Based on .*?, ', '', answer, flags=re.IGNORECASE) 
    answer = re.sub(r'^The (document|policy|contract) (states|mentions|indicates|specifies) that ', '', answer, flags=re.IGNORECASE)
    answer = re.sub(r'^From the document, ', '', answer, flags=re.IGNORECASE)
    answer = re.sub(r'^As per the (policy|document), ', '', answer, flags=re.IGNORECASE)
    
    # Clean up citations but preserve important bracketed information
    answer = re.sub(r'\(Page \d+\)', '', answer)
    answer = re.sub(r'\(Source:.*?\)', '', answer)
    
    # Keep the first complete paragraph/sentence group
    paragraphs = answer.split('\n')
    if paragraphs:
        answer = paragraphs[0].strip()
    
    # For accuracy, allow slightly longer answers but ensure completeness
    sentences = re.split(r'(?<=[.!?])\s+', answer)
    if sentences and sentences[0].strip():
        first_sentence = sentences[0].strip()
        
        # If first sentence seems incomplete (no period, very short), try to include more
        if (not first_sentence.endswith('.') and not first_sentence.endswith('!') and 
            not first_sentence.endswith('?') and len(sentences) > 1):
            # Combine first two sentences if it makes sense
            if len(first_sentence.split()) < 15 and len(sentences) > 1:
                answer = f"{first_sentence}. {sentences[1].strip()}"
            else:
                answer = first_sentence + '.'
        else:
            answer = first_sentence
    
    # Quality check: ensure answer has substance
    words = answer.split()
    if len(words) < 3:
        return "Information not available in the provided document."
    
    # Standardize number formatting (enhance for insurance terms)
    answer = re.sub(r'\b(\d+)\s*days?\b', r'\1 days', answer, flags=re.IGNORECASE)
    answer = re.sub(r'\b(\d+)\s*months?\b', r'\1 months', answer, flags=re.IGNORECASE)
    answer = re.sub(r'\b(\d+)\s*years?\b', r'\1 years', answer, flags=re.IGNORECASE)
    
    # Clean up extra spaces and ensure proper ending
    answer = re.sub(r'\s+', ' ', answer).strip()
    if not answer.endswith(('.', '!', '?')):
        answer += '.'
    
    return answer

async def process_document_and_answer(doc_url: str, questions: list[str]) -> list[str]:
    """
    Main pipeline: For a document and list of questions, returns answers.
    Optimized for parallel processing of document parsing and question answering.
    No caching - direct processing only.
    """
    start_time = time.time()
    logger.info(f"Processing document: {doc_url}")
    logger.info(f"Number of questions: {len(questions)}")
    
    # Process all questions directly (no caching)
    questions_to_process = [(i, q) for i, q in enumerate(questions)]
    cached_answers = {}
    
    # Start all parsing tasks in parallel
    parsing_task = download_and_parse_document(doc_url)
    
    # Wait for parsing to complete
    result = await parsing_task
    
    # Check if result is a tuple with text and meta
    if isinstance(result, tuple) and len(result) == 2:
        text, meta = result
    else:
        # Handle case where result might be a dict or other type
        if isinstance(result, dict):
            # This is the error case we're fixing - result came back as a dict
            logger.error("Document parsing returned a dict instead of (text, meta) tuple")
            # Try to extract text and meta from the dict
            text = result.get('text', '')
            meta = result.get('meta', {})
        else:
            text = str(result)
            meta = {}
    
    if not text:
        raise ValueError("Document parsing failed or empty document.")
    logger.info(f"Document parsed in {time.time() - start_time:.2f}s")
    
    # Step 2: Chunk text - optimized for semantic boundaries (now async)
    # Ensure text is a string before passing to chunk_text
    if not isinstance(text, str):
        logger.error(f"Expected text to be a string, but got {type(text)}")
        text = str(text)
        
    chunk_start = time.time()
    chunks, chunk_refs = await chunk_text(
        text, 
        meta, 
        chunk_size=config.CHUNK_SIZE, 
        overlap_size=config.OVERLAP_SIZE
    )
    logger.info(f"Document chunked into {len(chunks)} segments in {time.time() - chunk_start:.2f}s")
    
    # Step 3: Get or create embeddings asynchronously
    embedding_start = time.time()
    doc_id, embeddings = await get_or_create_embeddings(doc_url, chunks, chunk_refs)
    logger.info(f"Embeddings processed in {time.time() - embedding_start:.2f}s")
    
    # Step 4: Enhanced question processing with better context selection
    query_start = time.time()
    
    if questions_to_process:
        # Enhanced critical terms mapping using configuration
        critical_terms = {
            "grace_period": ["grace period", "grace", "renewal", "premium payment", "due date", "continuity", "policy period", "lapse"],
            "waiting_period": ["waiting period", "waiting", "pre-existing", "diseases", "coverage", "months", "years", "condition"],
            "maternity": ["maternity", "childbirth", "delivery", "female", "pregnancy", "natal", "birth"],
            "ayush": ["ayush", "ayurveda", "yoga", "naturopathy", "unani", "siddha", "homeopathy", "alternative"],
            "room_rent": ["room rent", "room charges", "accommodation", "icu", "intensive care"],
            "exclusions": ["excluded", "not covered", "exclusion", "does not cover", "limitation"],
            "hospital": ["hospital", "medical facility", "nursing home", "clinic"],
            "organ_donor": ["organ", "donor", "transplant", "donation"]
        }
        
        # Smart question processing - maintain speed but improve accuracy
        question_analysis = []
        for i, q in questions_to_process:
            q_lower = q.lower()
            
            # Enhanced question type detection for better context selection
            question_type = "general"
            priority = 1.0
            
            if any(term in q_lower for term in ["grace", "period", "payment", "premium", "due"]):
                question_type = "grace_period"
                priority = 1.5
            elif any(term in q_lower for term in ["waiting", "months", "pre-existing", "condition"]):
                question_type = "waiting_period"
                priority = 1.5
            elif any(term in q_lower for term in ["maternity", "pregnancy", "childbirth"]):
                question_type = "maternity"
                priority = 1.5
            elif any(term in q_lower for term in ["excluded", "not covered", "exclusion", "limitation"]):
                question_type = "exclusions"
                priority = 1.5
            elif any(term in q_lower for term in ["room", "rent", "charges", "icu"]):
                question_type = "room_rent"
                priority = 1.5
            elif any(term in q_lower for term in ["ayush", "ayurveda", "alternative"]):
                question_type = "ayush"
                priority = 1.5
            
            question_analysis.append({
                'index': i,
                'question': q,
                'type': question_type,
                'priority': priority
            })
        
        # Smart retrieval with adaptive top_k based on question importance
        retrieval_tasks = []
        for analysis in question_analysis:
            i, q = analysis['index'], analysis['question']
            # Use more chunks for high-priority questions without slowing down
            adaptive_top_k = 5 if analysis['priority'] > 1.0 else 3
            task = retrieve_similar_chunks_async(doc_id, q, chunks, chunk_refs, embeddings, adaptive_top_k)
            retrieval_tasks.append((analysis, task))
        
        # Execute all retrieval tasks in parallel
        retrieval_results = await asyncio.gather(*[task for _, task in retrieval_tasks])
        
        # Enhanced context processing for accuracy
        context_results = {}
        for (analysis, _), (top_chunks, top_refs) in zip(retrieval_tasks, retrieval_results):
            i, q = analysis['index'], analysis['question']
            question_type = analysis['type']
            
            # For high-priority questions, add one relevant chunk if available
            if analysis['priority'] > 1.0 and len(top_chunks) < 5:
                # Quick scan for additional relevant chunks
                q_keywords = set(q.lower().split())
                for idx, chunk in enumerate(chunks):
                    if chunk not in top_chunks:
                        chunk_keywords = set(chunk.lower().split())
                        # If chunk has 3+ matching keywords, include it
                        if len(q_keywords & chunk_keywords) >= 3:
                            top_chunks.append(chunk)
                            top_refs.append(chunk_refs[idx] if idx < len(chunk_refs) else "")
                            break  # Only add one additional chunk for speed
            
            # Ensure we have the best chunks in order
            if len(top_chunks) > 4:
                top_chunks = top_chunks[:4]
                top_refs = top_refs[:4]
            
            context_results[i] = (q, top_chunks, top_refs)
        
        # Enhanced batch processing for LLM calls
        prompts_data = []
        for i, (q, top_chunks, top_refs) in context_results.items():
            prompt = optimize_prompt(q, top_chunks, top_refs)
            prompts_data.append((i, prompt))
        
        # Process all prompts efficiently
        prompts = [data[1] for data in prompts_data]
        indices = [data[0] for data in prompts_data]
        
        if USE_OLLAMA:
            # For Ollama, process with controlled concurrency
            semaphore = asyncio.Semaphore(4)  # Limit concurrent requests
            
            async def process_single_ollama(q, top_chunks, top_refs):
                async with semaphore:
                    return await answer_with_llm(q, top_chunks, top_refs)
            
            llm_tasks = [
                process_single_ollama(q, top_chunks, top_refs)
                for i, (q, top_chunks, top_refs) in context_results.items()
            ]
            llm_results = await asyncio.gather(*llm_tasks)
        else:
            # For Cerebras, use optimized batch processing
            from app.llm import batch_cerebras_completions
            llm_results_batch = await batch_cerebras_completions(prompts)
            
            # Rearrange results back to original order
            llm_results = []
            for idx in sorted(indices):
                pos = indices.index(idx)
                llm_results.append(llm_results_batch[pos])
        
        # Process and store results
        for (i, (q, _, _)), answer in zip(context_results.items(), llm_results):
            # Enhanced post-processing
            answer = post_process_answer(answer)
            cached_answers[i] = answer
    
    # Prepare final answers in the original order
    answers = [cached_answers.get(i) for i in range(len(questions))]
    
    logger.info(f"Questions answered in {time.time() - query_start:.2f}s")
    logger.info(f"Total processing time: {time.time() - start_time:.2f}s")
    
    return answers
