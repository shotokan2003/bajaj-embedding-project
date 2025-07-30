import requests
import tempfile
from PyPDF2 import PdfReader
import docx
import os
from typing import List, Dict



def fetch_and_parse(url: str) -> Dict:
    """
    Fetches a document from a URL, parses content depending on file type,
    and splits text into chunks for downstream processing.
    
    Returns dictionary:
    {
        "chunks": List[{"text": str, "section_id": int}]
    }
    """
    # Identify extension for parsing
    ext = url.split('.')[-1].split('?')[0].lower()
    
    # Download to temp file
    response = requests.get(url)
    response.raise_for_status()
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{ext}') as tmp_file:
        tmp_file.write(response.content)
        tmp_path = tmp_file.name

    # Parse based on extension
    if ext == 'pdf':
        text = parse_pdf(tmp_path)
    elif ext == 'docx':
        text = parse_docx(tmp_path)
    elif ext in ['eml', 'msg']:  # Basic email parsing could be added (optional)
        text = parse_email(tmp_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    
    # Clean up temp file
    os.remove(tmp_path)
    
    # Chunk text into ~300 word chunks
    chunks = chunk_text(text, chunk_size=300)
    
    return {"chunks": chunks}

def parse_pdf(path: str) -> str:
    """Extract plain text from a PDF file using PyPDF2."""
    reader = PdfReader(path)
    text = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text.append(page_text)
    return "\n".join(text)

def parse_docx(path: str) -> str:
    """Extract plain text from a DOCX file using python-docx."""
    doc = docx.Document(path)
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n".join(paragraphs)

def parse_email(path: str) -> str:
    """Basic email parsing to get plain text body - can be improved as needed."""
    from email import policy
    from email.parser import BytesParser
    
    with open(path, 'rb') as f:
        msg = BytesParser(policy=policy.default).parse(f)
    body = msg.get_body(preferencelist=('plain'))
    if body:
        return body.get_content()
    return ""

def chunk_text(text: str, chunk_size: int = 300) -> List[Dict]:
    """
    Splits the text into chunks of approximately `chunk_size` words.
    Each chunk is a dict with text and section_id.
    """
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size):
        chunk_words = words[i:i + chunk_size]
        chunk_text = " ".join(chunk_words)
        chunks.append({
            "text": chunk_text,
            "section_id": i // chunk_size
        })
    return chunks
