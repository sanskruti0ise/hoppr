from backend.vector_db.vector_db_client import store_summary, retrieve_context


# -----------------------------
# Test data
# -----------------------------
conversation_id = "test_001"
role = "user"
content = "Hello, I need some help with my account."
summary = "User asked for account help."
flags = ["urgent", "account"]
llm_used = "gpt-4"

# -----------------------------
# Store summary
# -----------------------------
print("[Test] Storing summary in Weaviate...")
store_summary(conversation_id, role, content, summary, flags, llm_used)
print("[Test] Stored successfully!")

# -----------------------------
# Retrieve context
# -----------------------------
query = "account help"
print(f"[Test] Retrieving context for query: '{query}'...")
results = retrieve_context(query, top_k=3)

if results:
    print("[Test] Retrieved results:")
    for i, r in enumerate(results, 1):
        print(f"{i}. Summary: {r.get('summary')}, Flags: {r.get('flags')}, LLM: {r.get('llm_used')}")
else:
    print("[Test] No results found.")
