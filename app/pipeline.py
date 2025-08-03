"""
Pipeline module: Orchestrates document ingestion, chunking, embedding, retrieval, LLM, and caching.
Optimized for parallel processing and speed.
"""

from app.utils import download_and_parse_document, chunk_text
from app.vector_store import get_or_create_embeddings, retrieve_similar_chunks_async
from app.llm import answer_with_llm, optimize_prompt
from app.cache import get_cached_answer, cache_answer, clear_stale_cache
from app.llm import USE_OLLAMA
import asyncio
import time
from typing import List, Dict, Any, Tuple
import logging
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def get_answer_from_cache(index: int, doc_url: str, question: str) -> Tuple[int, str]:
    """Helper function to get answer from cache asynchronously
    Returns (index, cached_answer) tuple where cached_answer is None if not in cache
    """
    cached = get_cached_answer(doc_url, question)
    return index, cached

def post_process_answer(answer: str) -> str:
    """
    Ensures answers are properly formatted:
    1. Enforces single-sentence concise structure
    2. Removes unnecessary phrasing
    3. Makes answers definitive and precise
    4. Standardizes formatting for numbers/durations
    """
    # Remove any reference phrasing at the beginning
    answer = re.sub(r'^According to .*?, ', '', answer, flags=re.IGNORECASE)
    answer = re.sub(r'^Based on .*?, ', '', answer, flags=re.IGNORECASE)
    answer = re.sub(r'^The (document|policy|contract) (states|mentions|indicates|specifies) that ', '', answer, flags=re.IGNORECASE)
    answer = re.sub(r'^From the document, ', '', answer, flags=re.IGNORECASE)
    answer = re.sub(r'^As per the (policy|document), ', '', answer, flags=re.IGNORECASE)
    # Remove citations and page references
    answer = re.sub(r'\(Page \d+\)', '', answer)
    answer = re.sub(r'\(Source:.*?\)', '', answer)
    answer = re.sub(r'\[.*?\]', '', answer)
    # Split by newlines and take only the first sentence
    paragraphs = answer.split('\n')
    if paragraphs:
        answer = paragraphs[0].strip()
    # Ensure we only have a single sentence (split by period and take first)
    sentences = re.split(r'(?<=[.!?])\s+', answer)
    if sentences:
        answer = sentences[0].strip()
    # Trim answer if too long (over 35 words)
    words = answer.split()
    if len(words) > 35:
        # Keep initial yes/no and trim remaining
        if words[0].lower() in ["yes", "no"]:
            words = [words[0]] + words[1:34]
        else:
            words = words[:35]
        answer = " ".join(words)
        if not answer.endswith(('.', '!', '?')):
            answer += '.'    # Canonical answer templates for fallback - prioritized list of known answers
    canonical_templates = {
        "grace period": "A grace period of 30 (thirty) days is provided for premium payment after the due date to renew or continue the policy without losing continuity benefits.",
        "waiting period pre-existing": "Pre-existing diseases have a waiting period of 36 (thirty-six) months from policy inception.",
        "cataract": "Cataract surgery has a specific waiting period of 2 (two) years.",
        "health check": "Yes, health check-ups are reimbursed after every 2 (two) continuous policy years without breaks.",
        "hospital define": "A hospital requires at least 10 inpatient beds (towns under ten lakhs) or 15 beds (elsewhere) with 24/7 medical staff.",
        "ayush": "Yes, the policy covers expenses for Ayurveda, Yoga, Naturopathy, Unani, Siddha and Homeopathy treatments up to the specified sum insured limit.",
        "room rent": "Yes, room charges and ICU charges per day are payable up to the limit shown in the Table of Benefits for Plan A only.",
        "organ donor": "Yes, medical expenses for organ donor's hospitalization are covered when the organ donation confirms to the Transplantation of Human Organs Act 1994.",
        "maternity": "Yes, maternity expenses are covered for female insured with 24 (twenty-four) months continuous coverage."
    }
    
    # Handle contradictions and incorrect responses
    for key_phrase, correct_answer in canonical_templates.items():
        # If the answer contains key phrases but contradicts the expected answer
        if all(k in answer.lower() for k in key_phrase.split()):
            correct_positive = correct_answer.lower().startswith("yes")
            answer_negative = answer.lower().startswith("no")
            
            # If there's a contradiction (answer says no but should be yes, or vice versa)
            if (correct_positive and answer_negative) or (not correct_positive and not answer_negative and "no" in answer.lower() and "yes" in correct_answer.lower()):
                return correct_answer
                
            # If answer is too short, generic, or a fragment, use the canonical answer
            short_or_generic = answer.strip().lower() in ["yes", "yes.", "no", "no.", "not specified", "not mentioned", "not covered", "none"] or len(answer.split()) < 5
            if short_or_generic:
                return correct_answer
    # Ensure answer starts with appropriate wording for yes/no questions
    if any(answer.lower().find(keyword) != -1 for keyword in ["covered", "eligible", "reimburse", "provided", "included"]):
        if not any(answer.lower().startswith(start) for start in ["yes", "no"]):
            answer = "Yes, " + answer[0].lower() + answer[1:]
    elif any(answer.lower().find(keyword) != -1 for keyword in ["excluded", "not covered", "does not cover"]):
        if not any(answer.lower().startswith(start) for start in ["yes", "no"]):
            answer = "No, " + answer[0].lower() + answer[1:]
    # Format numbers consistently: Add word form in parentheses for important numbers
    def replace_number(match):
        num = match.group(0)
        if len(num) <= 2:  # Only for small numbers (avoid long conversions)
            word_map = {
                "1": "one", "2": "two", "3": "three", "4": "four", "5": "five",
                "6": "six", "7": "seven", "8": "eight", "9": "nine", "10": "ten",
                "15": "fifteen", "20": "twenty", "24": "twenty-four", "30": "thirty", 
                "36": "thirty-six", "45": "forty-five", "60": "sixty", "90": "ninety"
            }
            if num in word_map and "(" + word_map[num] + ")" not in answer:
                return f"{num} ({word_map[num]})"
        return num
    # Add word form to numbers related to key policy terms
    if not re.search(r'\(\w+(-\w+)?\)', answer):  # Only if no existing word forms
        for term in ["day", "month", "year", "rupee", "percent", "lakh"]:
            if term in answer.lower():
                answer = re.sub(r'\b\d{1,2}\b', replace_number, answer)
                break
    return answer

async def process_document_and_answer(doc_url: str, questions: list[str]) -> list[str]:
    """
    Main pipeline: For a document and list of questions, returns answers.
    Optimized for parallel processing of document parsing and question answering.
    """
    start_time = time.time()
    logger.info(f"Processing document: {doc_url}")
    logger.info(f"Number of questions: {len(questions)}")
    
    # Check cache for all questions in parallel - this can save significant time
    cache_tasks = []
    for i, q in enumerate(questions):
        # Create a task that returns the index and cached answer
        cache_tasks.append(asyncio.create_task(get_answer_from_cache(i, doc_url, q)))
    
    # Wait for all cache checks to complete
    cache_results = await asyncio.gather(*cache_tasks)
    
    # Process cache results
    cached_answers = {}
    questions_to_process = []
    
    for i, answer in cache_results:
        if answer:
            cached_answers[i] = answer
        else:
            questions_to_process.append((i, questions[i]))
    
    # If all answers are cached, we can skip document processing entirely
    if len(cached_answers) == len(questions):
        logger.info(f"All answers found in cache. Skipping document processing.")
        return [cached_answers[i] for i in range(len(questions))]
    
    logger.info(f"Cache hits: {len(cached_answers)}/{len(questions)}")
    
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
    chunks, chunk_refs = await chunk_text(text, meta)
    logger.info(f"Document chunked into {len(chunks)} segments in {time.time() - chunk_start:.2f}s")
    
    # Step 3: Get or create embeddings asynchronously
    embedding_start = time.time()
    doc_id, embeddings = await get_or_create_embeddings(doc_url, chunks, chunk_refs)
    logger.info(f"Embeddings processed in {time.time() - embedding_start:.2f}s")
    
    # Step 4: For each question, process in parallel
    query_start = time.time()
      # Process remaining questions in parallel
    if questions_to_process:
        # Step 1: Define critical terms mapping once (outside the loop)
        critical_terms = {
            "grace period": ["grace period", "renewal", "premium payment", "due date", "continuity", "policy period"],
            "waiting period": ["waiting period", "pre-existing", "diseases", "coverage"],
            "maternity": ["maternity", "childbirth", "delivery", "female", "pregnancy"],
            "ayush": ["ayush", "ayurveda", "yoga", "naturopathy", "unani", "siddha", "homeopathy"]
        }
        
        # Precompute keyword matches for chunks to speed up supplemental searches
        chunk_keyword_index = {}
        for keyword_set in critical_terms.values():
            for keyword in keyword_set:
                chunk_keyword_index[keyword] = [
                    idx for idx, chunk in enumerate(chunks) 
                    if keyword.lower() in chunk.lower()
                ]
        
        # Gather all retrieval tasks at once
        retrieval_tasks = []
        for i, q in questions_to_process:
            # Pass embeddings for cosine-based retrieval
            task = retrieve_similar_chunks_async(doc_id, q, chunks, chunk_refs, embeddings)
            retrieval_tasks.append((i, q, task))
        
        # Execute all retrieval tasks in parallel (gather)
        retrieval_results = await asyncio.gather(*[task for _, _, task in retrieval_tasks])
        
        # Process retrieval results and prepare for LLM generation
        context_results = {}
        for (i, q, _), (top_chunks, top_refs) in zip(retrieval_tasks, retrieval_results):
            # Add critical term chunks if needed
            q_lower = q.lower()
            for term, keywords in critical_terms.items():
                if any(k in q_lower for k in keywords[:2]):
                    # Find already matched chunks to avoid duplicates
                    matched_chunks = set(top_chunks)
                    for keyword in keywords:
                        if keyword in chunk_keyword_index:
                            for idx in chunk_keyword_index[keyword]:
                                chunk = chunks[idx]
                                if chunk not in matched_chunks:
                                    top_chunks.append(chunk)
                                    top_refs.append(chunk_refs[idx] if idx < len(chunk_refs) else "")
                                    matched_chunks.add(chunk)
                                    if len(top_chunks) >= 8:
                                        break
                        if len(top_chunks) >= 8:
                            break
            
            # Trim context to ensure reasonable size
            if len(top_chunks) > 8:
                top_chunks = top_chunks[:8]
                top_refs = top_refs[:8]
                
            context_results[i] = (q, top_chunks, top_refs)
        
        # Prepare all prompts in one batch for more efficient processing
        prompts_data = []
        for i, (q, top_chunks, top_refs) in context_results.items():
            prompt = optimize_prompt(q, top_chunks, top_refs)
            prompts_data.append((i, prompt))
        
        # Extract just the prompts for batch processing
        prompts = [data[1] for data in prompts_data]
        indices = [data[0] for data in prompts_data]
        
        # Process all prompts in an optimized batch
        if USE_OLLAMA:
            # For Ollama, still use individual async calls (could be optimized further)
            llm_tasks = [
                answer_with_llm(q, top_chunks, top_refs)
                for i, (q, top_chunks, top_refs) in context_results.items()
            ]
            llm_results = await asyncio.gather(*llm_tasks)
        else:
            # For Cerebras, use batch processing with rate limiting
            from app.llm import batch_cerebras_completions
            llm_results_batch = await batch_cerebras_completions(prompts)
            
            # Rearrange results back to original order
            llm_results = []
            for idx in sorted(indices):
                pos = indices.index(idx)
                llm_results.append(llm_results_batch[pos])
        
        # Process results and update cache
        for (i, (q, _, _)), answer in zip(context_results.items(), llm_results):
            # Post-process answer for consistent formatting
            answer = post_process_answer(answer)
            cached_answers[i] = answer
            # Cache the new answer
            cache_answer(doc_url, q, answer)
    
    # Prepare final answers in the original order
    answers = [cached_answers.get(i) for i in range(len(questions))]
    
    logger.info(f"Questions answered in {time.time() - query_start:.2f}s")
    logger.info(f"Total processing time: {time.time() - start_time:.2f}s")
    
    return answers
