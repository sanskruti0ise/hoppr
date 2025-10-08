from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from backend.router.mistral_router import route_chat
from backend.summarizer.mistral_summarizer import summarize_chat
from backend.vector_db.vector_db_client import store_summary, retrieve_context

# -----------------------------
# Pydantic models
# -----------------------------
class Message(BaseModel):
    role: str
    content: str

class HopprRequest(BaseModel):
    conversation_id: str  # new field for session tracking
    messages: List[Message]

class HopprResponse(BaseModel):
    routing_decision: dict
    summary: Optional[str] = None
    key_points: Optional[List[str]] = None
    flags: Optional[List[str]] = None
    raw: Optional[str] = None

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(title="Hoppr Backend API", version="0.6.0")

# Enable CORS for frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or specify frontend URL(s)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Existing routes
# -----------------------------
@app.get("/")
def home():
    return {"message": "Welcome to Hoppr 🚀"}

@app.post("/route")
def route(chat: dict):
    messages = chat.get("messages", [])
    routing_decision = route_chat(messages)
    return {"routing_decision": routing_decision}

@app.post("/summarize")
def summarize_endpoint(payload: HopprRequest):
    result = summarize_chat([m.dict() for m in payload.messages])
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["message"])

    # Clean nested summary
    if isinstance(result.get("summary"), dict):
        result = result["summary"]

    return {
        "summary": result.get("summary"),
        "key_points": result.get("key_points"),
        "flags": result.get("flags"),
        "raw": result.get("raw")
    }

# -----------------------------
# 🚀 Updated Hoppr pipeline with Weaviate
# -----------------------------
@app.post("/hoppr", response_model=HopprResponse, tags=["Hoppr Pipeline"])
def hoppr_pipeline(payload: HopprRequest):
    """
    Full Hoppr pipeline:
    1️⃣ Retrieve past relevant context from vector DB
    2️⃣ Route the conversation to the next LLM
    3️⃣ Summarize conversation
    4️⃣ Store new summary in vector DB
    """
    messages = [m.dict() for m in payload.messages]

    # Step 1: Retrieve relevant context from Weaviate
    last_message = messages[-1]["content"] if messages else ""
    context_objects = retrieve_context(last_message, top_k=3)
    context_summaries = [obj["summary"] for obj in context_objects if "summary" in obj]

    # Prepend retrieved summaries as system messages
    if context_summaries:
        messages = [{"role": "system", "content": "\n".join(context_summaries)}] + messages

    # Step 2: routing decision
    routing_decision = route_chat(messages)

    # Step 3: conversation summarization
    summary_result = summarize_chat(messages)
    if "error" in summary_result:
        raise HTTPException(status_code=500, detail=summary_result["message"])
    if isinstance(summary_result.get("summary"), dict):
        summary_result = summary_result["summary"]

    # Step 4: store summary in Weaviate
    store_summary(
        conversation_id=payload.conversation_id,
        role="assistant",
        content=last_message,
        summary=summary_result.get("summary"),
        flags=summary_result.get("flags"),
        llm_used=routing_decision.get("next_llm", "unknown")
    )

    return {
        "routing_decision": routing_decision,
        "summary": summary_result.get("summary"),
        "key_points": summary_result.get("key_points"),
        "flags": summary_result.get("flags"),
        "raw": summary_result.get("raw")
    }
