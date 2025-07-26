import os
from groq import Groq

client = Groq(api_key=os.environ["GROQ_API_KEY"])

def generate_answer_from_chunks(question: str, chunks: list) -> str:
    context = "\n\n".join(chunks)
    prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer concisely based on the context above."
    response = client.chat.completions.create(
        model="llama3-70b-8192",  # or "llama-3.3-70b-versatile" per Groq docs
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
        temperature=0,
    )
    return response.choices[0].message.content.strip()
