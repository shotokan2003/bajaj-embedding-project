# app/utils.py
from PyPDF2 import PdfReader

def parse_pdf(path):
    """
    Extract plain text from PDF file.
    :param path: str file path
    :return: (text:str, sections:list) Sections metadata empty for now
    """
    reader = PdfReader(path)
    text = "\n".join(page.extract_text() for page in reader.pages if page.extract_text())
    sections = []  # Placeholder for future clause/section extraction
    return text, sections

def parse_docx(path):
    """
    Extract plain text from DOCX file.
    :param path: str file path
    :return: (text:str, sections:list)
    """
    import docx
    doc = docx.Document(path)
    text = "\n".join(p.text for p in doc.paragraphs)
    sections = []  # TODO: Enhance with real section splitting
    return text, sections

def parse_email(path):
    """
    Extract plain text from .eml or .msg email files.
    :param path: str filepath
    :return: (text:str, sections:list)
    """
    from email import policy
    from email.parser import BytesParser
    with open(path, "rb") as f:
        msg = BytesParser(policy=policy.default).parse(f)
    text = msg.get_body(preferencelist=('plain')).get_content()
    sections = []
    return text, sections

def chunk_by_section(text, sections):
    """
    Splits text into 300-word chunks with section mapping if present.
    :param text: str full text
    :param sections: list of sections metadata (unused, empty)
    :return: List[Dict] {"text":..., "section_id":...}
    """
    words = text.split()
    chunk_size = 500  # words per chunk
    chunks = []
    
    for i in range(0, len(words), chunk_size):
        chunk_text = " ".join(words[i:i+chunk_size])
        chunks.append({
            "text": chunk_text,
            "section_id": i // chunk_size
        })
    return chunks
