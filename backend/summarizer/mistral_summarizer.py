import os
import re
import json
from dotenv import load_dotenv
from mistralai import Mistral
from mistralai.models import SDKError

load_dotenv()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
if not MISTRAL_API_KEY:
    raise RuntimeError("Please set MISTRAL_API_KEY in environment or .env")

client = Mistral(api_key=MISTRAL_API_KEY)

def clean_json_output(text: str):
    """
    Remove code fences and parse JSON if possible.
    Return dict or fallback with raw text.
    """
    # Remove fences like ```json … ``` or ``` … ```
    cleaned = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    cleaned = cleaned.strip("`").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {"raw": text}

def summarize_chat(messages: list[dict], model: str = "mistral-large-latest", temperature: float = 0.3):
    """
    messages: list of dicts, each with keys 'role' and 'content'.
      e.g. [{"role":"user","content":"..."}]
    Returns a dict: either parsed JSON summary or fallback.
    """
    # Ensure the last message is from user (Mistral requirement)
    if not messages:
        return {"error": "No messages provided"}

    # If the last message is from assistant, drop it
    if messages[-1].get("role") == "assistant":
        messages = messages[:-1]

    # Prompt for summarization
    system_prompt = (
        "You are Hoppr's summarization brain. "
        "Summarize the conversation in JSON format with keys: summary (short text), "
        "key_points (list of important bullets), flags (optional tags)."
    )
    # Build prompt sequence
    prompt_msgs = [
        {"role": "system", "content": system_prompt},
    ]
    # Append conversation
    prompt_msgs.extend(messages)

    try:
        resp = client.chat.complete(
            model=model,
            messages=prompt_msgs,
            temperature=temperature,
        )
    except SDKError as e:
        # handle API error
        return {
            "error": "Mistral SDK error",
            "status_code": getattr(e, "status_code", None),
            "message": str(e),
        }

    # Extract response text
    choice = resp.choices[0]
    content = choice.message.content if hasattr(choice.message, "content") else choice.message["content"]
    # Clean and parse JSON output
    parsed = clean_json_output(content)
    return parsed
