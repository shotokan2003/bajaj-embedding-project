import os
from groq import Groq
from app.cache import get_answer_from_redis, set_answer_to_redis
import asyncio

client = Groq(api_key=os.environ["GROQ_API_KEY"])

def generate_answer_from_chunks(question: str, chunks: list) -> str:
    # Try Redis cache (async)
    loop = asyncio.get_event_loop()
    redis_answer = loop.run_until_complete(get_answer_from_redis(question))
    if redis_answer:
        return redis_answer
    context = "\n\n".join(chunks)
    prompt = (
        "You are an expert assistant.\n"
        "Answer the following question using ONLY the provided context.\n"
        "If the answer is not present in the context, reply with 'I don't know.'\n"
        "Be concise, accurate, and clear.\n"
        "\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
    ).format(context=context, question=question)
    response = client.chat.completions.create(
        model="llama3-70b-8192",  # or "llama-3.3-70b-versatile" per Groq docs
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
        temperature=0,
    )
    answer = response.choices[0].message.content.strip()
    # Set in Redis (async)
    loop.run_until_complete(set_answer_to_redis(question, answer))
    return answer
