"""
Utils: Download, parse, and chunk documents with parallel processing.
"""

import requests
import fitz  # PyMuPDF
import docx
import hashlib
import asyncio
import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Tuple, Any
from app.cache import get_cached_document, cache_document

MAX_WORKERS = min(32, os.cpu_count() + 4)  # Optimal thread count

def hash_str(s: str) -> str:
    """Returns a SHA256 hash of a string."""
    return hashlib.sha256(s.encode()).hexdigest()

async def download_and_parse_document(url: str) -> Tuple[str, Dict[str, Any]]:
    """
    Downloads and parses a PDF or DOCX document from a URL with caching.
    Returns (text, meta) where meta includes page/section info.
    Uses in-memory parsing to avoid temp files.
    """
    # Check cache first
    cached = get_cached_document(url)
    if cached:
        # Validate the cache format to ensure it's a tuple with (text, meta)
        if isinstance(cached, tuple) and len(cached) == 2:
            text, meta = cached
            # Ensure text is a string and meta is a dict and text is not empty
            if isinstance(text, str) and isinstance(meta, dict) and text.strip():
                return text, meta
        # If cached text is empty or format is invalid, force re-parse
        import logging
        logging.warning(f"Invalid or empty cache for {url}. Forcing re-parse.")
        # Continue to re-parse below

    # Use aiohttp for async HTTP requests
    import aiohttp
    import io
    
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            if resp.status != 200:
                raise ValueError("Failed to download document.")
            
            content_type = resp.headers.get("content-type", "")
            content = await resp.read()
            meta = {}
    
    # Process document based on type
    if ".pdf" in url or "pdf" in content_type:
        # Parse PDF in-memory with parallel processing
        loop = asyncio.get_event_loop()
        
        # Open PDF directly from memory buffer
        with ThreadPoolExecutor() as executor:
            doc = await loop.run_in_executor(
                executor,
                lambda: fitz.open(stream=content, filetype="pdf")
            )
        
        num_pages = len(doc)
        meta["total_pages"] = num_pages
        
        # Process pages in parallel
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            tasks = [
                loop.run_in_executor(
                    executor, 
                    extract_page_text, 
                    doc, i
                )
                for i in range(num_pages)
            ]
            page_results = await asyncio.gather(*tasks)
        
        # Combine results
        text = ""
        page_refs = []
        
        for i, (page_text, page_tables) in enumerate(page_results):
            if page_text.strip():
                text += f"--- PAGE {i+1} ---\n"
                text += page_text + "\n"
                page_refs.append((len(text), f"Page {i+1}"))
                
                # Add table data if available
                if page_tables:
                    for t_idx, table in enumerate(page_tables):
                        text += f"[TABLE {t_idx+1}]\n"
                        text += table + "\n"
        
        meta["page_refs"] = page_refs
        
        # Close the document in a thread to avoid blocking
        with ThreadPoolExecutor() as executor:
            await loop.run_in_executor(executor, doc.close)
        
    elif ".docx" in url or "word" in content_type:
        # Parse DOCX directly from memory
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            def parse_docx_from_memory():
                doc = docx.Document(io.BytesIO(content))
                return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
            
            text = await loop.run_in_executor(executor, parse_docx_from_memory)
        
        meta["page_refs"] = []
        
    else:
        raise ValueError("Unsupported document type.")
    
    # Ensure text is valid and not empty
    if not text.strip():
        raise ValueError("Document parsing failed or empty document.")
    
    # Cache only if text is valid (don't await to avoid blocking)
    asyncio.create_task(async_cache_document(url, text, meta))
    return text, meta

async def async_cache_document(url: str, text: str, meta: Dict[str, Any]):
    """Async wrapper to cache document without blocking"""
    cache_document(url, text, meta)

def extract_page_text(doc: fitz.Document, page_num: int) -> Tuple[str, List[str]]:
    """Extract text and tables from a page (used for parallel processing)"""
    page = doc[page_num]
    page_text = page.get_text()
    
    # Extract tables if available (simplified)
    tables = []
    try:
        for table in page.find_tables():
            table_text = ""
            for row in table.rows:
                row_texts = [cell.text for cell in row.cells]
                table_text += " | ".join(row_texts) + "\n"
            tables.append(table_text)
    except:
        # Table extraction is experimental in PyMuPDF
        pass
        
    return page_text, tables

async def chunk_text(text: str, meta: dict, chunk_size: int = 800) -> tuple[list[str], list[str]]:
    """
    Splits text into semantic chunks (paragraphs, sections) up to chunk_size words.
    Returns chunks and their references.
    
    Uses smaller chunk size (800 vs 1000) and better boundary detection to improve retrieval accuracy.
    Async version for better performance.
    """
    # Handle case where text might not be a string
    if not isinstance(text, str):
        import logging
        logging.error(f"Expected text to be a string, got {type(text)}. Converting to string.")
        text = str(text)
    # First, identify key insurance policy sections to preserve intact
    critical_policy_sections = {
        "grace period": r'(?i)(\bgrace period\b.*?(?:\.|$)(?:[^\n]*\n?){0,3})',
        "waiting period": r'(?i)(\bwaiting period\b.*?(?:\.|$)(?:[^\n]*\n?){0,3})',
        "pre-existing": r'(?i)(\bpre-existing disease.*?(?:\.|$)(?:[^\n]*\n?){0,3})',
        "maternity": r'(?i)(\bmaternity.*?(?:\.|$)(?:[^\n]*\n?){0,3})',
        "ayush": r'(?i)(\bayush.*?(?:\.|$)(?:[^\n]*\n?){0,3})',
        "room rent": r'(?i)(\broom rent.*?(?:\.|$)(?:[^\n]*\n?){0,3})',
        "renewal": r'(?i)(\brenewal.*?(?:\.|$)(?:[^\n]*\n?){0,3})',
    }
    
    # Extract and save critical sections to ensure they're preserved
    preserved_sections = []
    for topic, pattern in critical_policy_sections.items():
        matches = re.findall(pattern, text)
        for match in matches:
            if len(match) > 20:  # Only preserve non-trivial matches
                preserved_sections.append((match, topic))
    
    # Split by semantic boundaries with improved pattern
    # Include section headers, page markers, paragraph breaks, bullet points
    semantic_splits = re.split(
        r'(?:\n\n+|---|^#{1,3}\s+|\[TABLE.*?\]|\n\d+\.\s+|\n[A-Z]\.\s+|\n•\s+)',
        text
    )
    
    chunks = []
    refs = []
    
    current_chunk = []
    current_chunk_size = 0
    current_ref = ""
    page_refs = meta.get("page_refs", [])
      
    for split in semantic_splits:
        split = split.strip()
        if not split:
            continue
            
        # Find page reference for this split
        for cutoff, page in page_refs:
            if text.find(split) < cutoff:
                ref = page
                break
        else:
            ref = ""
        
        split_words = split.split()
        
        # If adding this split would make chunk too big, create a new chunk
        if current_chunk_size + len(split_words) > chunk_size and current_chunk:
            chunks.append(" ".join(current_chunk))
            refs.append(current_ref)
            current_chunk = []
            current_chunk_size = 0
            
        # Add this split to the current chunk
        current_chunk.extend(split_words)
        current_chunk_size += len(split_words)
        if ref and not current_ref:  # Only update ref if we don't have one yet
            current_ref = ref
    
    # Add the final chunk if it exists
    if current_chunk:
        chunks.append(" ".join(current_chunk))
        refs.append(current_ref)
    
    # Now add the preserved critical sections as additional chunks
    # This ensures important policy clauses appear in their own chunks for better retrieval
    for section_text, topic in preserved_sections:
        # Find page reference for this section
        for cutoff, page in page_refs:
            if text.find(section_text) < cutoff:
                ref = page
                break
        else:
            ref = ""
        
        # Only add if not a duplicate (exact match) of an existing chunk
        if section_text not in chunks:
            # Add context around section for better understanding
            enriched_section = f"Policy section about {topic}: {section_text}"
            chunks.append(enriched_section)
            refs.append(ref)
    
    return chunks, refs
