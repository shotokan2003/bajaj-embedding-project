import os
import asyncio
import json
import re
from groq import Groq

# Initialize Groq client
_groq = Groq(api_key=os.getenv("GROQ_API_KEY"))

FEW_SHOT_EXAMPLES = """
Example 1:
Q: What is the grace period for premium payment?
A: A grace period of 30 (thirty) days is allowed for premium payment after the due date to avoid policy lapse.

Example 2:
Q: Does this policy cover maternity expenses?
A: Yes, maternity expenses are covered if the insured has 24 (twenty-four) months continuous coverage, limited to two deliveries or terminations.
"""

def clean_and_parse_llm_output(text):
    """
    Remove markdown code fences and parse JSON answers from the LLM output.
    Return a list containing only the answer strings.
    """
    # Remove markdown code block fences (``````json etc)
    cleaned = re.sub(r"``````+", "", text, flags=re.DOTALL).strip()

    try:
        data = json.loads(cleaned)

        if isinstance(data, list):
            answers = []
            for item in data:
                if isinstance(item, dict) and "answer" in item:
                    answers.append(item["answer"])
                else:
                    answers.append(str(item))
            return answers
        else:
            return [text.strip()]

    except json.JSONDecodeError as e:
        print(f"JSON parse error in LLM output: {e}")
        return [text.strip()]

async def batch_extract_answers_with_retry(questions, context, max_retries=3):
    """
    Batch all questions and submit to Groq LLM with retries on rate-limiting.

    Returns:
        List[str]: List of answer strings in order corresponding to questions
    """
    questions_text = "\n".join([f"{i + 1}. {q}" for i, q in enumerate(questions)])

    prompt = f"""
You are a precise and reliable insurance policy assistant.

Answer the questions ONLY using the information provided in the CONTEXT.
If the information is not present, respond with: "Not specified in the provided context."

Answer each question in a concise, complete sentence, 25-30 words max.
Use exact numerical values and terms from the CONTEXT.

Do NOT use vague language or fillers like "according to the document."

Provide your answers as a JSON array of objects with "question" and "answer" fields matching the question order.

Here are a few examples:

{FEW_SHOT_EXAMPLES}

---

CONTEXT:
{context}

QUESTIONS:
{questions_text}

Respond ONLY with the JSON array of answers.
"""

    retries = 0
    wait_time = 1.0  # seconds

    while True:
        try:
            completion = _groq.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=512,
                temperature=0.1,
            )

            choice = completion.choices[0]
            if isinstance(choice, dict) and "message" in choice:
                answer_text = choice["message"]["content"]
            elif hasattr(choice, "message") and hasattr(choice.message, "content"):
                answer_text = choice.message.content
            elif isinstance(choice, dict) and "text" in choice:
                answer_text = choice["text"]
            else:
                answer_text = str(choice)

            answers = clean_and_parse_llm_output(answer_text)

            if len(answers) != len(questions):
                print("WARNING: Number of answers differs from number of questions.")

            return answers

        except Exception as e:
            if ("rate_limit" in str(e).lower() or "429" in str(e)) and retries < max_retries:
                print(f"Rate limit hit. Retrying in {wait_time}s (attempt {retries + 1} of {max_retries})...")
                await asyncio.sleep(wait_time)
                wait_time *= 2
                retries += 1
            else:
                raise e
