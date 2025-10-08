import os
import json
import re
from dotenv import load_dotenv
from mistralai import Mistral

load_dotenv()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

if not MISTRAL_API_KEY:
    raise ValueError("⚠️ MISTRAL_API_KEY is missing from .env")

client = Mistral(api_key=MISTRAL_API_KEY)

def clean_and_parse_json(text: str):
    """
    Clean markdown-style code fences and parse JSON safely.
    """
    # remove code fences like ```json ... ```
    cleaned = re.sub(r"^```json|```$", "", text, flags=re.MULTILINE).strip()
    cleaned = cleaned.strip("`").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # fallback: return as raw string
        return {"raw": text, "error": "failed to parse json"}

def route_chat(messages):
    """
    Use Mistral to analyze chat and return structured routing decision.
    """
    system_prompt = (
        "You are Hoppr's routing brain. "
        "Analyze the conversation and respond with a JSON object like this:\n"
        "{ \"next_llm\": \"gpt-4 or claude or mistral\", \"context\": \"short summary\", \"flags\": [\"math\", \"coding\"] }"
    )

    chat_messages = [{"role": "system", "content": system_prompt}] + messages

    response = client.chat.complete(
        model="mistral-large-latest",
        messages=chat_messages,
        temperature=0.2,
    )

    result_text = response.choices[0].message.content
    return clean_and_parse_json(result_text)
