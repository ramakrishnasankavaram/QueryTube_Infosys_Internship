import sys
from pathlib import Path
import numpy as np

DB_PERSIST_DIR = "chroma_db"
COLLECTION_NAME = "youtube_videos"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 5

def embed_query(query, model_name=EMBED_MODEL_NAME):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    emb = model.encode([query], convert_to_numpy=True)[0]
    return emb.tolist()

def create_chroma_client(persist_directory):
    """Create ChromaDB client compatible with version 0.4.0+"""
    import chromadb
    
    persist_path = Path(persist_directory)
    if not persist_path.exists():
        raise FileNotFoundError(
            f"ChromaDB directory '{persist_directory}' not found. "
            "Please run vector_db.py first to create the database."
        )
    
    # Try PersistentClient first (ChromaDB 0.4.0+)
    try:
        client = chromadb.PersistentClient(path=str(persist_path))
        print(f"✓ Connected to ChromaDB at '{persist_path}'")
        return client
    except AttributeError:
        print("PersistentClient not available, trying alternatives...")
    except Exception as ex:
        print(f"PersistentClient failed: {ex}")
    
    # Try HttpClient for server-based setup
    try:
        client = chromadb.HttpClient()
        print("✓ Connected via HttpClient")
        return client
    except Exception:
        pass
    
    # Fallback: try Client with path (some intermediate versions)
    try:
        client = chromadb.Client(path=str(persist_path))
        print(f"✓ Connected with Client(path=...)")
        return client
    except Exception as ex:
        print(f"Client(path=...) failed: {ex}")
    
    raise RuntimeError(
        "Failed to create ChromaDB client.\n"
        "Please upgrade ChromaDB: pip install --upgrade chromadb\n"
        "Required version: 0.4.0 or higher"
    )

def main(query=None, top_k=TOP_K):
    if query is None:
        query = input("Enter your search query: ").strip()
    
    if not query:
        print("No query provided. Exiting.")
        return
    
    print(f"Embedding query: '{query}'")
    q_emb = embed_query(query)

    # Use the new client creation function
    client = create_chroma_client(DB_PERSIST_DIR)
    
    try:
        collection = client.get_collection(name=COLLECTION_NAME)
        print(f"✓ Loaded collection: '{COLLECTION_NAME}'")
    except Exception as e:
        raise RuntimeError(
            f"Failed to get collection '{COLLECTION_NAME}'. Error: {e}\n"
            "Make sure you've run vector_db.py first."
        )

    # Query - NOTE: 'ids' are ALWAYS returned, don't include in 'include' parameter
    results = collection.query(
        query_embeddings=[q_emb], 
        n_results=top_k, 
        include=["metadatas", "documents", "distances"]
    )
    
    ids = results.get("ids", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]
    documents = results.get("documents", [[]])[0]

    # Chroma returns distances depending on metric; commonly lower is more similar for cosine/inner
    # We'll convert distance to similarity with sim = 1 - distance (works if distances in [0,1]).
    print("\n" + "="*80)
    print(f"Top {len(ids)} results for: '{query}'")
    print("="*80 + "\n")
    
    if not ids:
        print("No results found.")
        return
    
    for i, vid in enumerate(ids):
        meta = metadatas[i] if i < len(metadatas) else {}
        dist = distances[i] if i < len(distances) else None
        sim = None
        if dist is not None:
            try:
                sim = 1.0 - float(dist)
            except Exception:
                sim = None
        title = meta.get("title") or ""
        channel = meta.get("channel_title") or ""
        url = f"https://www.youtube.com/watch?v={vid}"
        
        print(f"{i+1}. Video ID: {vid}")
        print(f"   Title: {title}")
        print(f"   Channel: {channel}")
        print(f"   URL: {url}")
        print(f"   Similarity: {sim:.4f}" if sim is not None else f"   Similarity: N/A")
        print(f"   Raw Distance: {dist}")
        print()

if __name__ == '__main__':
    qry = None
    if len(sys.argv) > 1:
        qry = " ".join(sys.argv[1:])
    main(query=qry)