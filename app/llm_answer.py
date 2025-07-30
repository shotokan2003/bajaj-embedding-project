# app/llm_answer.py
import os
from groq import Groq

# Initialize Groq client with API key from environment variable
_groq = Groq(api_key=os.getenv("GROQ_API_KEY"))  # Make sure GROQ_API_KEY is set in your environment

def extract_answers(question, top_chunks):
    """
    Given a question and top retrieved text chunks, call Groq LLM to generate answer & rationale
    :param question: str
    :param top_chunks: List[Dict] with text chunks and metadata
    :return: (answer: str, rationale: str)
    """
    # Concatenate texts for context feeding into LLM
    context = "\n\n".join([c["text"] for c in top_chunks])
    
    # Construct the carefully designed prompt as requested
    prompt = (
        "Answer the insurance policy question using ONLY the provided document excerpts. "
        "YOUR ANSWER MUST BE A SINGLE, COMPLETE SENTENCE, MAXIMUM 25-30 WORDS, NEVER JUST 'YES', 'NO', OR A FRAGMENT. "
        "If the answer is 'yes' or 'no', ALWAYS provide a brief explanation in the same sentence. "
        "DO NOT use phrases like 'According to the document', 'Based on', or 'The policy states'. "
        "NEVER say 'not specified' if information exists in the document. "
        "STATE FACTS DIRECTLY using the exact numbers, durations, limits and conditions. "
        "Format numbers both as digits and words in parentheses: e.g. '30 (thirty)' days. "
        "Avoid connectors like 'additionally', 'furthermore', 'moreover'. "
        "DO NOT leave the answer incomplete or cut off.\n\n"
        "EXAMPLES:\n"
        "Question: What is the waiting period for pre-existing diseases?\n"
        "BAD: According to the document, pre-existing diseases have a waiting period of 36 months from policy inception.\n"
        "GOOD: Pre-existing diseases have a waiting period of 36 (thirty-six) months from policy inception.\n\n"
        "Question: Is there a grace period for premium payment?\n"
        "BAD: There is no grace period.\n"
        "GOOD: A grace period of 30 (thirty) days is provided for premium payment after the due date to renew or continue the policy without losing continuity benefits.\n\n"
        "Question: Does the policy cover maternity expenses?\n"
        "BAD: The policy mentions coverage for maternity expenses subject to certain conditions like continuous coverage.\n"
        "GOOD: Yes, maternity expenses are covered for female insured with 24 (twenty-four) months continuous coverage.\n\n"
        "Question: How are day care procedures handled?\n"
        "BAD: The policy provides coverage for various day care procedures that don't require 24 hours hospitalization, as mentioned in the policy documents.\n"
        "GOOD: Day care procedures not requiring 24-hour hospitalization are covered as specified in the policy annexure.\n\n"
        "CONTEXT:\n" + context + "\n\n"
        "QUESTION: " + question + "\n"
        "ANSWER:"
    )
    
    # Call Groq chat completion API
    completion = _groq.chat.completions.create(
        model="llama-3.1-8b-instant",  # Replace with your valid model if different
        messages=[{"role": "user", "content": prompt}],
        max_tokens=256,
        temperature=0.1
    )
    
    answer_text = completion.choices[0].message.content
    
    # Optionally: split answer and rationale if model outputs both
    parts = answer_text.split("Rationale:", 1)
    answer = parts[0].strip()
    rationale = parts[1].strip() if len(parts) > 1 else "Rationale included in answer text."
    
    return answer, rationale
