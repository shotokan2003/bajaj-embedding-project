import re
import requests
import tempfile
from PyPDF2 import PdfReader
import docx
import os
from typing import List, Dict

def fetch_and_parse(url: str) -> Dict:
    ext = url.split('.')[-1].split('?').lower()
    response = requests.get(url)
    response.raise_for_status()

    with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{ext}') as tmp_file:
        tmp_file.write(response.content)
        tmp_path = tmp_file.name

    if ext == 'pdf':
        text = parse_pdf(tmp_path)
    elif ext == 'docx':
        text = parse_docx(tmp_path)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

    os.remove(tmp_path)
    chunks = chunk_by_sections(text)
    return {"chunks": chunks}

def parse_pdf(path: str) -> str:
    reader = PdfReader(path)
    texts = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            texts.append(page_text)
    return "\n".join(texts)

def parse_docx(path: str) -> str:
    document = docx.Document(path)
    paragraphs = [p.text for p in document.paragraphs if p.text.strip()]
    return "\n".join(paragraphs)

def chunk_by_sections(text: str, min_chunk_words=150, max_chunk_words=350) -> List[Dict]:
    pattern = r'\n(?=Section\s+\d+|Clause\s+\d+(\.\d+)?|^\d+\.\s|^[A-Z][A-Za-z\s]{3,}\n)'
    sections = re.split(pattern, text, flags=re.MULTILINE)

    if len(sections) < 2:
        return chunk_text(text, chunk_size=300)

    chunks = []
    for idx, sec in enumerate(sections):
        words = sec.split()
        for i in range(0, len(words), max_chunk_words):
            chunk_words = words[i:i + max_chunk_words]
            chunk_text = " ".join(chunk_words)
            if len(chunk_words) >= min_chunk_words:
                chunks.append({
                    "text": chunk_text,
                    "section_id": idx
                })
    return chunks

def chunk_text(text: str, chunk_size: int = 300) -> List[Dict]:
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append({"text": chunk, "section_id": i // chunk_size})
    return chunks
