import os
import time
import atexit
import warnings
import weaviate
from weaviate.classes.config import Property, DataType, Configure
from sentence_transformers import SentenceTransformer

# ======================================================
# Optional: Silence Deprecation warnings (for cleaner test logs)
# ======================================================
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ======================================================
# Weaviate connection setup
# ======================================================
WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://weaviate:8080")
host = WEAVIATE_URL.replace("http://", "").replace("https://", "").split(":")[0]

client = None
max_retries = 10
retry_delay = 3  # seconds

for attempt in range(1, max_retries + 1):
    try:
        client = weaviate.connect_to_custom(
            http_host=host,
            http_port=8080,
            http_secure=False,
            grpc_host=host,
            grpc_port=50051,
            grpc_secure=False,
        )

        meta = client.get_meta()
        version = (
            meta.get("version")
            if isinstance(meta, dict)
            else getattr(meta, "version", "unknown")
        )
        hostname = (
            meta.get("hostname", host)
            if isinstance(meta, dict)
            else getattr(meta, "hostname", host)
        )

        print(
            f"[VectorDB] ✅ Connected to Weaviate at {WEAVIATE_URL} "
            f"({hostname}, version: {version})"
        )
        break

    except Exception as e:
        print(
            f"[VectorDB] ⚠️ Attempt {attempt}/{max_retries} - "
            f"Failed to connect to Weaviate at {WEAVIATE_URL}: {e}"
        )
        if attempt == max_retries:
            print("[VectorDB] ❌ Max retries reached. Exiting.")
            raise e
        time.sleep(retry_delay)

# Ensure connection closes on exit
atexit.register(lambda: client and client.close())

# ======================================================
# Embedding model
# ======================================================
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# ======================================================
# Schema setup
# ======================================================
class_name = "ConversationSummary"

try:
    existing_classes = client.collections.list_all()
    # Handle both string and object responses across SDK versions
    if existing_classes and hasattr(existing_classes[0], "name"):
        existing_class_names = [col.name for col in existing_classes]
    else:
        existing_class_names = existing_classes or []
except Exception as e:
    print(f"[VectorDB] ⚠️ Could not list collections: {e}")
    existing_class_names = []

if class_name not in existing_class_names:
    try:
        client.collections.create(
            name=class_name,
            description="Stores Hoppr conversation summaries and routing info",
            vectorizer_config=Configure.Vectorizer.none(),
            properties=[
                Property(name="conversation_id", data_type=DataType.TEXT),
                Property(name="role", data_type=DataType.TEXT),
                Property(name="content", data_type=DataType.TEXT),
                Property(name="summary", data_type=DataType.TEXT),
                Property(name="flags", data_type=DataType.TEXT_ARRAY),
                Property(name="llm_used", data_type=DataType.TEXT),
            ],
        )
        print(f"[VectorDB] 🏗️ Created collection '{class_name}' in Weaviate.")
    except Exception as e:
        if "already exists" in str(e):
            print(f"[VectorDB] ⚠️ Collection '{class_name}' already exists, continuing.")
        else:
            raise
else:
    print(f"[VectorDB] ℹ️ Using existing collection '{class_name}'.")

# Retrieve collection reference
collection = client.collections.get(class_name)

# ======================================================
# Store summary in Weaviate
# ======================================================
def store_summary(conversation_id, role, content, summary, flags, llm_used):
    """Store a conversation summary with its embedding in Weaviate."""
    try:
        vector = embed_model.encode(summary)
        collection.data.insert(
            properties={
                "conversation_id": conversation_id,
                "role": role,
                "content": content,
                "summary": summary,
                "flags": flags,
                "llm_used": llm_used,
            },
            vector=vector.tolist(),
        )
        print(f"[VectorDB] ✅ Stored summary for conversation_id='{conversation_id}'")
    except Exception as e:
        print(f"[VectorDB] ❌ Failed to store summary: {e}")

# ======================================================
# Retrieve relevant context
# ======================================================
def retrieve_context(query, top_k=3):
    """Retrieve top-k summaries most similar to the query."""
    try:
        vector = embed_model.encode(query)
        response = collection.query.near_vector(
            near_vector=vector.tolist(),
            limit=top_k,
        )

        results = [
            {
                "summary": obj.properties.get("summary"),
                "flags": obj.properties.get("flags"),
                "llm_used": obj.properties.get("llm_used"),
            }
            for obj in response.objects
        ]

        print(f"[VectorDB] 🔍 Retrieved {len(results)} results for query: '{query}'")
        return results

    except Exception as e:
        print(f"[VectorDB] ❌ Retrieval failed: {e}")
        return []

# ======================================================
# Manual close function (optional)
# ======================================================
def close_connection():
    """Safely close the Weaviate client connection."""
    try:
        if client is not None:
            client.close()
            print("[VectorDB] 🔒 Connection to Weaviate closed.")
    except Exception as e:
        print(f"[VectorDB] ⚠️ Error closing Weaviate connection: {e}")
