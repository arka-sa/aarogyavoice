"""
VoiceDoc - Qdrant Knowledge Base Ingestion
Run this ONCE to set up your Qdrant collection with medical knowledge.
Usage: python ingest.py
"""

import json
import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

load_dotenv()

COLLECTION_NAME = "voicedoc_medical"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Fast, free, no API key needed
VECTOR_SIZE = 384  # Dimension for all-MiniLM-L6-v2


def main():
    print("VoiceDoc - Qdrant Ingestion Script")
    print("=" * 40)

    # Connect to Qdrant
    print("\n[1/4] Connecting to Qdrant Cloud...")
    client = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
    )
    print(f"      Connected: {os.getenv('QDRANT_URL')}")

    # Load embedding model
    print("\n[2/4] Loading embedding model (first run downloads ~90MB)...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    print(f"      Model ready: {EMBEDDING_MODEL}")

    # Create or recreate collection
    print(f"\n[3/4] Setting up Qdrant collection: '{COLLECTION_NAME}'...")
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)
        print("      Deleted old collection.")

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )
    print("      Collection created.")

    # Load and ingest medical knowledge
    print("\n[4/4] Ingesting medical knowledge base...")
    with open("data/medical_kb.json", "r") as f:
        documents = json.load(f)

    texts = [doc["text"] for doc in documents]
    embeddings = model.encode(texts, show_progress_bar=True)

    points = [
        PointStruct(
            id=i,
            vector=embeddings[i].tolist(),
            payload={
                "text": doc["text"],
                "category": doc["category"],
                "doc_id": doc["id"],
            },
        )
        for i, doc in enumerate(documents)
    ]

    client.upsert(collection_name=COLLECTION_NAME, points=points)

    print(f"\n✅ Done! Ingested {len(points)} medical knowledge entries into Qdrant.")
    print(f"   Collection: '{COLLECTION_NAME}'")
    print(f"\nNext step: copy .env.example to .env, fill in your keys, then run:")
    print("   uvicorn main:app --reload --port 8000")


if __name__ == "__main__":
    main()
