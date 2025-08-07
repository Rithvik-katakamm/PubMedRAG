# Test 4: Save this as test_components.py
import sys
sys.path.insert(0, 'src')

# Test OpenAI connection
print("=== Testing OpenAI Connection ===")
try:
    from openai import OpenAI
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "hello"}],
        max_tokens=10
    )
    print("✅ OpenAI: SUCCESS")
    print(f"Response: {response.choices[0].message.content}")
except Exception as e:
    print(f"❌ OpenAI: FAILED - {e}")

# Test PubMed retrieval
print("\n=== Testing PubMed Retrieval ===")
try:
    from core.retrieval import fetch_pubmed_abstracts
    ids, text = fetch_pubmed_abstracts("test@example.com", "diabetes", retmax=2)
    print(f"✅ PubMed: SUCCESS - Got {len(ids)} IDs, {len(text)} chars")
except Exception as e:
    print(f"❌ PubMed: FAILED - {e}")

# Test ChromaDB
print("\n=== Testing ChromaDB ===")
try:
    from core.vector_db import ChromaDBManager
    db = ChromaDBManager()
    stats = db.get_stats()
    print(f"✅ ChromaDB: SUCCESS - {len(stats)} collections")
except Exception as e:
    print(f"❌ ChromaDB: FAILED - {e}")

# Test Embeddings
print("\n=== Testing Embeddings ===")
try:
    from core.embeddings import EmbeddingGenerator
    embedder = EmbeddingGenerator()
    embeddings = embedder.embed_chunks(["test sentence"])
    print(f"✅ Embeddings: SUCCESS - {len(embeddings[0])} dimensions")
except Exception as e:
    print(f"❌ Embeddings: FAILED - {e}")