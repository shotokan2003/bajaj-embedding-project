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
from app.performance import monitor_performance
from app.config import config
import logging

logger = logging.getLogger(__name__)

MAX_WORKERS = min(32, os.cpu_count() + 4)  # Optimal thread count

def hash_str(s: str) -> str:
    """Returns a SHA256 hash of a string."""
    return hashlib.sha256(s.encode()).hexdigest()

@monitor_performance("document_parsing")
async def download_and_parse_document(url: str) -> Tuple[str, Dict[str, Any]]:
    """
    Downloads and parses a PDF or DOCX document from a URL directly (no caching).
    Returns (text, meta) where meta includes page/section info.
    Uses in-memory parsing to avoid temp files.
    """
    import logging
    logging.info(f"Parsing document from {url}")
    # No cache checking - always parse directly

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
        
        # Process pages in parallel with reduced workers for faster processing
        optimal_workers = min(8, MAX_WORKERS, num_pages)  # Reduced from MAX_WORKERS
        with ThreadPoolExecutor(max_workers=optimal_workers) as executor:
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
    
    # No caching - return directly
    return text, meta

def extract_page_text(doc: fitz.Document, page_num: int) -> Tuple[str, List[str]]:
    """Extract text and tables from a page with simplified processing for speed"""
    page = doc[page_num]
    
    # Use simple text extraction for speed - skip complex structure detection
    page_text = page.get_text()
    
    # Only extract tables if they're simple to find
    tables = []
    try:
        # Simplified table extraction - timeout after 1 second per page
        import signal
        def timeout_handler(signum, frame):
            raise TimeoutError("Table extraction timeout")
        
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(1)  # 1 second timeout
        
        for table in page.find_tables():
            table_text = "\n[TABLE START]\n"
            for row in table.rows[:5]:  # Limit to first 5 rows for speed
                row_texts = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                if row_texts:
                    table_text += " | ".join(row_texts) + "\n"
            table_text += "[TABLE END]\n"
            tables.append(table_text)
            if len(tables) >= 3:  # Limit number of tables per page
                break
        signal.alarm(0)  # Clear alarm
    except (Exception, TimeoutError):
        # Skip table extraction if it takes too long or fails
        pass
        
    return page_text, tables

@monitor_performance("text_chunking")
async def chunk_text(text: str, meta: dict, chunk_size: int = 1000, overlap_size: int = 50) -> tuple[list[str], list[str]]:
    """
    Ultra-fast chunking with larger chunks to minimize embedding calls.
    
    Args:
        text: Document text
        meta: Document metadata 
        chunk_size: Target chunk size in words (increased for fewer chunks)
        overlap_size: Minimal overlap for speed
    
    Returns:
        chunks and their references
    """
    # Handle case where text might not be a string
    if not isinstance(text, str):
        text = str(text)
    
    # Enhanced critical section extraction for better accuracy
    critical_keywords = {
        "grace_period": ["grace period", "premium payment", "due date"],
        "waiting_period": ["waiting period", "pre-existing", "months"],
        "maternity": ["maternity", "pregnancy", "childbirth"],
        "exclusions": ["not covered", "excluded", "exclusion", "limitation"],
        "room_rent": ["room rent", "room charges", "accommodation"],
        "ayush": ["ayush", "ayurveda", "alternative medicine"]
    }
    
    # Smart critical section extraction with better context
    preserved_sections = []
    text_lower = text.lower()
    sentences = text.split('.')
    
    for topic, keywords in critical_keywords.items():
        for keyword in keywords:
            if keyword in text_lower:
                # Find sentences with context (previous and next sentence)
                for i, sentence in enumerate(sentences):
                    if keyword in sentence.lower() and len(sentence.strip()) > 20:
                        # Build context with surrounding sentences
                        context_sentences = []
                        if i > 0:  # Add previous sentence for context
                            context_sentences.append(sentences[i-1].strip())
                        context_sentences.append(sentence.strip())
                        if i < len(sentences) - 1:  # Add next sentence for context
                            context_sentences.append(sentences[i+1].strip())
                        
                        context = '. '.join(context_sentences) + '.'
                        if len(context) > 80:  # Ensure meaningful context
                            preserved_sections.append((context, topic.replace('_', ' ')))
                        break
                break  # Only one per topic for speed
    
    # Minimal boundary detection for speed
    sections = re.split(r'\n={3,}.*?\n|\n\n\n+', text)
    sections = [s.strip() for s in sections if s.strip()]
    
    # Create large chunks efficiently
    chunks = []
    refs = []
    page_refs = meta.get("page_refs", [])
    
    def get_page_ref(text_pos: int) -> str:
        if not page_refs:
            return ""
        # Fast lookup - just use first match
        for cutoff, page in page_refs:
            if text_pos < cutoff:
                return page
        return ""
    
    # Process sections into large chunks with minimal overlap
    for section in sections:
        if not section.strip():
            continue
            
        words = section.split()
        if len(words) <= chunk_size:
            chunks.append(section)
            text_pos = text.find(section[:30])
            refs.append(get_page_ref(text_pos))
        else:
            # Large chunks with minimal overlap for speed
            for i in range(0, len(words), max(1, chunk_size - overlap_size)):
                chunk_words = words[i:i + chunk_size]
                if len(chunk_words) < 50:  # Skip small chunks
                    break
                    
                chunk_text = " ".join(chunk_words)
                chunks.append(chunk_text)
                text_pos = text.find(chunk_text[:30])
                refs.append(get_page_ref(text_pos))
                
                if i + chunk_size >= len(words):
                    break
    
    # Fast addition of critical sections
    existing_chunks_lower = {chunk.lower()[:50] for chunk in chunks}
    
    for section_text, topic in preserved_sections:
        section_key = section_text.lower()[:50]
        if section_key not in existing_chunks_lower:
            enhanced_section = f"[{topic.upper()}] {section_text}"
            chunks.append(enhanced_section)
            refs.append("Policy")
    
    # Aggressive chunk limit for speed
    max_chunks = min(config.MAX_CHUNKS, 60)  # Hard limit for ultra-fast processing
    if len(chunks) > max_chunks:
        logger.warning(f"Limiting chunks from {len(chunks)} to {max_chunks} for speed")
        chunks = chunks[:max_chunks]
        refs = refs[:max_chunks]
    
    return chunks, refs
