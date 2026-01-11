# utils.py
import os
from pathlib import Path
from typing import List, Optional
import numpy as np

# load env
from dotenv import load_dotenv
load_dotenv()

DB_PERSIST_DIR = os.getenv("DB_PERSIST_DIR", "chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "youtube_videos")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", None)
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# -------- Chroma client creation (robust) ----------
def create_chroma_client(persist_dir: Optional[str] = None):
    persist_dir = persist_dir or DB_PERSIST_DIR
    from pathlib import Path
    p = Path(persist_dir)
    p.mkdir(parents=True, exist_ok=True)
    import chromadb
    try:
        if hasattr(chromadb, "PersistentClient"):
            client = chromadb.PersistentClient(path=str(p))
            return client
    except Exception:
        pass
    try:
        from chromadb.config import Settings
        client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=str(p)))
        return client
    except Exception:
        pass
    try:
        client = chromadb.Client(path=str(p))
        return client
    except Exception:
        client = chromadb.Client()
        return client

def get_collection(client=None, name: Optional[str] = None):
    client = client or create_chroma_client()
    name = name or COLLECTION_NAME
    try:
        if hasattr(client, "get_collection"):
            return client.get_collection(name=name)
    except Exception:
        pass
    if hasattr(client, "get_or_create_collection"):
        return client.get_or_create_collection(name=name)
    if hasattr(client, "create_collection"):
        return client.create_collection(name=name)
    raise RuntimeError("Unable to get/create collection")

# ---------- Embeddings ----------
# Uses sentence-transformers locally to create embeddings for queries.
_embedding_model = None
def load_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    return _embedding_model

def embed_text(text: str) -> List[float]:
    model = load_embedding_model()
    emb = model.encode(text)
    # convert numpy to list
    return emb.tolist()

# ---------- Similarity search ----------
def query_top_k(query_text: str, k: int = 5, collection=None):
    # embed query
    q_emb = embed_text(query_text)
    client = create_chroma_client()
    coll = collection or get_collection(client)
    # Some Chroma APIs: query(query_embeddings=...), or query(query_texts=...)
    # NOTE: ChromaDB returns ids by default; do NOT include "ids" in the include parameter
    try:
        res = coll.query(query_embeddings=[q_emb], n_results=k, include=["metadatas", "documents", "distances"])
    except Exception as e:
        print(f"[query_top_k] query_embeddings failed: {e}, trying query_texts...")
        # fallback: use query_texts instead of embeddings
        try:
            res = coll.query(query_texts=[query_text], n_results=k, include=["metadatas", "documents", "distances"])
        except Exception as e2:
            print(f"[query_top_k] query_texts also failed: {e2}")
            raise e2
    
    # Parse results (Chroma returns nested lists)
    # res["ids"], res["documents"], res["metadatas"] are returned as list-of-lists
    # e.g., ids = [[id1, id2, ...]], documents = [[doc1, doc2, ...]], etc.
    ids_raw = res.get("ids", [[]])
    docs_raw = res.get("documents", [[]])
    metadatas_raw = res.get("metadatas", [[]])
    distances_raw = res.get("distances", [[]])
    
    # Extract first sublist if nested (this is the expected format)
    ids = ids_raw[0] if ids_raw and isinstance(ids_raw[0], list) else ids_raw
    docs = docs_raw[0] if docs_raw and isinstance(docs_raw[0], list) else docs_raw
    metadatas = metadatas_raw[0] if metadatas_raw and isinstance(metadatas_raw[0], list) else metadatas_raw
    distances = distances_raw[0] if distances_raw and isinstance(distances_raw[0], list) else distances_raw
    
    # Ensure all have same length
    n = min(len(ids), len(docs), len(metadatas)) if ids and docs and metadatas else 0
    ids = ids[:n]
    docs = docs[:n]
    metadatas = metadatas[:n]
    distances = distances[:n] if distances else [0.0] * n
    
    # Convert distances to similarity scores.
    # Chroma's cosine "distance" is derived from cosine similarity and can be in [0, 2]
    # where distance = 1 - cosine_similarity (so cosine_similarity = 1 - distance).
    # To map to a 0..1 similarity percentage we use: similarity = 1 - (distance / 2).
    similarities = []
    for d in distances:
        if d is None:
            # If distance is None, use a neutral score (50%) to avoid dropping the result
            similarities.append(0.5)
            continue
        try:
            dist_float = float(d)
        except (TypeError, ValueError):
            similarities.append(0.5)
            continue

        # Map distance (expected 0..2) to similarity (1..0)
        sim = 1.0 - (dist_float / 2.0)
        # Clamp to [0.0, 1.0]
        sim = max(0.0, min(1.0, sim))
        similarities.append(sim)

    # Debug: show distances and computed similarities for the top results
    dist_preview = [f"{float(d):.4f}" if d is not None else "None" for d in (distances[:5] if distances else [])]
    sim_preview = [f"{s:.4f}" for s in (similarities[:5] if similarities else [])]
    print(f"[query_top_k] Found {n} results for query: '{query_text}'. distances: {dist_preview} similarities: {sim_preview}")

    return list(zip(ids, metadatas, docs, similarities))

# ---------- Summarization (Gemini or local fallback) ----------
def summarize_with_gemini(text: str) -> str:
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY not set")
    from google import genai
    client = genai.Client(api_key=GEMINI_API_KEY)
    prompt = "You are an expert summarizer. Produce a concise 3-6 sentence summary:\n\n" + text
    response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
    if hasattr(response, "text"):
        return response.text.strip()
    return str(response).strip()

def summarize_local(text: str) -> str:
    # local transformers fallback (BART)
    from transformers import pipeline
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=-1)
    # chunk large text
    chunks = []
    max_len = 1000
    start = 0
    while start < len(text):
        chunks.append(text[start:start+max_len])
        start += max_len
    parts = []
    for c in chunks:
        out = summarizer(c, max_length=200, min_length=40, do_sample=False)
        parts.append(out[0]["summary_text"].strip())
    # combine and compress if necessary
    combined = " ".join(parts)
    if len(combined) > 1000:
        out = summarizer(combined, max_length=200, min_length=40, do_sample=False)
        return out[0]["summary_text"].strip()
    return combined

def summarize_text(text: str) -> str:
    try:
        return summarize_with_gemini(text)
    except Exception:
        return summarize_local(text)

# ---------- util: youtube thumbnail ----------
def youtube_thumbnail_url(video_id: str) -> str:
    return f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg"

def youtube_watch_url(video_id: str) -> str:
    return f"https://www.youtube.com/watch?v={video_id}"
