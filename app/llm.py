"""
LLM module: Handles LLM calls with optimized processing and error handling.
"""

import os
import requests
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Optional
import logging
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Import Cerebras SDK for fast inference
try:
    from cerebras.cloud.sdk import Cerebras
except ImportError:
    logger.warning("Cerebras SDK not installed. Run: pip install cerebras-cloud-sdk")

load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

# LLM configuration
USE_OLLAMA = os.getenv("USE_OLLAMA", "0") == "1"  # Use Ollama if specified
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

# Cerebras configuration - using llama-4-scout-17b for optimal speed and quality
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY", "")
CEREBRAS_MODEL = os.getenv("CEREBRAS_MODEL", "llama-4-scout-17b-16e-instruct") # Fast model with good token limit
# Enable debug logging for API requests
DEBUG_API = os.getenv("DEBUG_API", "0") == "1"

# Performance settings
MAX_RETRIES = 3
RETRY_BACKOFF = 1.5  # seconds, exponential
MAX_WORKERS = min(8, os.cpu_count() or 4)
REQUEST_TIMEOUT = 30  # seconds

class LLMError(Exception):
    """Error during LLM call"""
    pass

def _cerebras_completion(prompt: str) -> str:
    """Call Cerebras API with retry logic"""
    if not CEREBRAS_API_KEY:
        raise LLMError("CEREBRAS_API_KEY not set")
    
    # Initialize Cerebras client
    client = Cerebras(api_key=CEREBRAS_API_KEY)
    
    # Log the request if in debug mode
    if DEBUG_API:
        logger.info(f"Cerebras API request with prompt: {prompt[:100]}...")
    
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            # Create the chat completion request with optimized parameters
            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "user", "content": prompt}
                ],
                model=CEREBRAS_MODEL,
                temperature=0.05,  # Lower temperature for more consistent answers
                max_tokens=100,    # Reduced for faster generation and concise answers
                top_p=0.95,        # Focus on high-probability tokens
                stream=False       # Ensure non-streaming for consistency
            )
            
            # Process response
            return chat_completion.choices[0].message.content.strip()
            
        except Exception as e:
            error_message = f"Cerebras API error: {str(e)}"
                
            if attempt == MAX_RETRIES:
                logger.error(error_message)
                raise LLMError(error_message)
            
            wait_time = RETRY_BACKOFF ** attempt
            logger.warning(f"{error_message}, retrying in {wait_time}s")
            time.sleep(wait_time)
    
    raise LLMError("Failed to get response from Cerebras API")

async def _ollama_completion(prompt: str, model: str = OLLAMA_MODEL) -> str:
    """Call Ollama API with retry logic asynchronously"""
    import aiohttp
    
    data = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": 256,
            "temperature": 0.1,  # Lower temp for more deterministic answers
        }
    }
    
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    OLLAMA_URL, 
                    json=data,
                    timeout=REQUEST_TIMEOUT
                ) as resp:
                    resp.raise_for_status()
                    result = await resp.json()
                    return result.get("response", "").strip()
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            if attempt == MAX_RETRIES:
                raise LLMError(f"Ollama API error: {str(e)}")
            wait_time = RETRY_BACKOFF ** attempt
            logger.warning(f"Ollama API error: {str(e)}, retrying in {wait_time}s")
            await asyncio.sleep(wait_time)
    
    raise LLMError("Failed to get response from Ollama API")

def optimize_prompt(question: str, context_chunks: List[str], refs: List[str]) -> str:
    """Create an optimized prompt focusing on key context only"""
    # Combine chunks with their references
    context_items = []
    for i, (chunk, ref) in enumerate(zip(context_chunks, refs)):
        # Truncate very long chunks to essential parts
        if len(chunk) > 1000:
            # Extract sentences that might contain the answer
            sentences = chunk.split('. ')
            relevant = []
            question_words = set(question.lower().split())
            
            # Keep sentences with question keywords
            for sentence in sentences:
                sentence_words = set(sentence.lower().split())
                if len(question_words.intersection(sentence_words)) >= 2:
                    relevant.append(sentence)
            
            # If we found relevant sentences, use them
            if relevant:
                chunk = '. '.join(relevant[:5]) + '.'
        
        # Add reference if available
        prefix = f"[{ref}] " if ref else ""
        context_items.append(f"{prefix}{chunk}")
    
    # Join contexts with clear separation
    context = "\n\n".join(context_items)
    
    # Construct the final prompt with even stricter instructions
    prompt = (
        f"Answer the insurance policy question using ONLY the provided document excerpts. "
        f"YOUR ANSWER MUST BE A SINGLE, COMPLETE SENTENCE, MAXIMUM 25-30 WORDS, NEVER JUST 'YES', 'NO', OR A FRAGMENT. "
        f"If the answer is 'yes' or 'no', ALWAYS provide a brief explanation in the same sentence. "
        f"DO NOT use phrases like 'According to the document', 'Based on', or 'The policy states'. "
        f"NEVER say 'not specified' if information exists in the document. "
        f"STATE FACTS DIRECTLY using the exact numbers, durations, limits and conditions. "
        f"Format numbers both as digits and words in parentheses: e.g. '30 (thirty)' days. "
        f"Avoid connectors like 'additionally', 'furthermore', 'moreover'. "
        f"DO NOT leave the answer incomplete or cut off.\n\n"
        f"EXAMPLES:\n"
        f"Question: What is the waiting period for pre-existing diseases?\n"
        f"BAD: According to the document, pre-existing diseases have a waiting period of 36 months from policy inception.\n"
        f"GOOD: Pre-existing diseases have a waiting period of 36 (thirty-six) months from policy inception.\n\n"
        f"Question: Is there a grace period for premium payment?\n"
        f"BAD: There is no grace period.\n"
        f"GOOD: A grace period of 30 (thirty) days is provided for premium payment after the due date to renew or continue the policy without losing continuity benefits.\n\n"
        f"Question: Does the policy cover maternity expenses?\n"
        f"BAD: The policy mentions coverage for maternity expenses subject to certain conditions like continuous coverage.\n"
        f"GOOD: Yes, maternity expenses are covered for female insured with 24 (twenty-four) months continuous coverage.\n\n"
        f"Question: How are day care procedures handled?\n"
        f"BAD: The policy provides coverage for various day care procedures that don't require 24 hours hospitalization, as mentioned in the policy documents.\n"
        f"GOOD: Day care procedures not requiring 24-hour hospitalization are covered as specified in the policy annexure.\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION: {question}\n"
        f"ANSWER:"
    )
    
    return prompt

async def answer_with_llm(question: str, context_chunks: List[str], refs: List[str]) -> str:
    """
    Calls LLM with context and question, returns answer with clause reference.
    Optimized for performance with parallel processing using Cerebras.
    """
    # Create optimized prompt focusing on relevant context
    prompt = optimize_prompt(question, context_chunks, refs)
    
    if USE_OLLAMA:
        try:
            # Directly use the async function for Ollama
            return await _ollama_completion(prompt)
        except LLMError as e:
            logger.error(f"Ollama error: {str(e)}")
            # Fall back to Cerebras if Ollama fails and Cerebras key exists
            if CEREBRAS_API_KEY:
                logger.info("Falling back to Cerebras API")
                # Run synchronous Cerebras in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                    return await loop.run_in_executor(executor, _cerebras_completion, prompt)
            raise
    else:
        # Use Cerebras as the primary LLM (synchronous function in thread pool)
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            return await loop.run_in_executor(executor, _cerebras_completion, prompt)
