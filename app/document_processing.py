import httpx
import pdfplumber
import nltk
from typing import List

nltk.download('punkt')

async def download_pdf(url: str) -> bytes:
    async with httpx.AsyncClient() as client:
        resp = await client.get(url)
        resp.raise_for_status()
        return resp.content

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    with pdfplumber.open(pdf_bytes) as pdf:
        text = "\n\n".join(page.extract_text() or "" for page in pdf.pages)
    return text

def chunk_text(text: str, max_tokens=500) -> List[str]:
    # Simple paragraph splitting fallback
    paragraphs = text.split('\n\n')

    # Optional: You can further split/merge paragraphs based on length or semantic similarity
    # For now, just return paragraphs.
    return [p.strip() for p in paragraphs if p.strip()]

async def download_and_chunk(document_url: str) -> List[str]:
    pdf_bytes = await download_pdf(document_url)
    # pdfplumber requires a file-object, we use BytesIO
    import io
    pdf_file = io.BytesIO(pdf_bytes)
    text = extract_text_from_pdf(pdf_file)
    chunks = chunk_text(text)
    return chunks
