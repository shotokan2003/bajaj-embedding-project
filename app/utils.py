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
    """
    # Check cache first
    cached = get_cached_document(url)
    if cached:
        return cached
    
    resp = requests.get(url)
    if resp.status_code != 200:
        raise ValueError("Failed to download document.")
    
    content_type = resp.headers.get("content-type", "")
    meta = {}
    
    if ".pdf" in url or "pdf" in content_type:
        # Parse PDF with parallel processing
        with open("temp.pdf", "wb") as f:
            f.write(resp.content)
        
        doc = fitz.open("temp.pdf")
        num_pages = len(doc)
        meta["total_pages"] = num_pages
        
        # Process pages in parallel
        loop = asyncio.get_event_loop()
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
        doc.close()
        
    elif ".docx" in url or "word" in content_type:
        # Parse DOCX
        with open("temp.docx", "wb") as f:
            f.write(resp.content)
        doc = docx.Document("temp.docx")
        text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
        meta["page_refs"] = []
        
    else:
        raise ValueError("Unsupported document type.")
    
    # Cache the document
    cache_document(url, text, meta)
    return text, meta

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

def chunk_text(text: str, meta: dict, chunk_size: int = 1000) -> tuple[list[str], list[str]]:
    """
    Splits text into semantic chunks (paragraphs, sections) up to chunk_size words.
    Returns chunks and their references.
    """
    # Split by semantic boundaries
    # Look for section markers, paragraphs, page breaks etc.
    semantic_splits = re.split(r'(?:\n\n+|---|^#{1,3}\s+|\[TABLE.*?\])', text)
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
    
    return chunks, refs
