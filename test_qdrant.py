# test_qdrant.py - run this separately to diagnose
from qdrant_client import QdrantClient
from langchain_community.embeddings import HuggingFaceEmbeddings

qdrant_client = QdrantClient(url="http://localhost:6333")

# 1. Check collection exists and has points
try:
    info = qdrant_client.get_collection("financial_docs")
    print(f"Collection found!")
    print(f"Points count: {info.points_count}")
    print(f"Vector size: {info.config.params.vectors.size}")
    print(f"Distance: {info.config.params.vectors.distance}")
except Exception as e:
    print(f"Collection error: {e}")

# 2. Scroll through a few raw records to see payload structure
try:
    records, _ = qdrant_client.scroll(
        collection_name="financial_docs",
        limit=3,
        with_payload=True,
        with_vectors=False
    )
    print(f"\nSample records ({len(records)} shown):")
    for r in records:
        print(f"\n--- ID: {r.id} ---")
        print(f"Payload keys: {list(r.payload.keys())}")
        print(f"Payload preview: {str(r.payload)[:300]}")
except Exception as e:
    print(f"Scroll error: {e}")

# 3. Test a direct search
try:
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    query_vector = embeddings.embed_query("insurance policy")
    results = qdrant_client.search(
        collection_name="financial_docs",
        query_vector=query_vector,
        limit=3
    )
    print(f"\nSearch results: {len(results)} found")
    for r in results:
        print(f"Score: {r.score}, Payload keys: {list(r.payload.keys())}")
        print(f"Payload preview: {str(r.payload)[:300]}")
except Exception as e:
    print(f"Search error: {e}")