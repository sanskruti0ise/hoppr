import weaviate
from sentence_transformers import SentenceTransformer
import os

# -----------------------------
# Weaviate connection
# -----------------------------
WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://localhost:8080")

client = weaviate.Client(WEAVIATE_URL)

# -----------------------------
# Embedding model
# -----------------------------
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# -----------------------------
# Create schema if not exists
# -----------------------------
class_name = "ConversationSummary"
existing_classes = client.schema.get()["classes"]

if not any(c["class"] == class_name for c in existing_classes):
    schema = {
        "classes": [
            {
                "class": class_name,
                "description": "Stores Hoppr conversation summaries and routing info",
                "vectorizer": "none",  # we provide embeddings
                "properties": [
                    {"name": "conversation_id", "dataType": ["string"]},
                    {"name": "role", "dataType": ["string"]},
                    {"name": "content", "dataType": ["text"]},
                    {"name": "summary", "dataType": ["text"]},
                    {"name": "flags", "dataType": ["string[]"]},
                    {"name": "llm_used", "dataType": ["string"]},
                ],
            }
        ]
    }
    client.schema.create(schema)

# -----------------------------
# Store summary in Weaviate
# -----------------------------
def store_summary(conversation_id, role, content, summary, flags, llm_used):
    vector = embed_model.encode(summary)
    client.data_object.create(
        data_object={
            "conversation_id": conversation_id,
            "role": role,
            "content": content,
            "summary": summary,
            "flags": flags,
            "llm_used": llm_used,
        },
        class_name=class_name,
        vector=vector.tolist()
    )

# -----------------------------
# Retrieve relevant context
# -----------------------------
def retrieve_context(query, top_k=3):
    vector = embed_model.encode(query)
    result = (
        client.query.get(class_name, ["summary", "flags", "llm_used"])
        .with_near_vector({"vector": vector.tolist()})
        .with_limit(top_k)
        .do()
    )
    return result.get("data", {}).get("Get", {}).get(class_name, [])
