# app.py
"""
FastAPI app for ingesting vector DB records into Chroma.

Endpoints:
- POST /ingest/json  -> JSON payload (single record or list of records)
- POST /ingest/csv   -> multipart/form-data CSV upload (embedding column must be JSON-list or Python-list string)
- GET  /health       -> simple ok

Record format (JSON):
{
  "video_id": "doZ4Q2ULR30",            # optional - if missing an id will be generated
  "title": "Example title",
  "transcript": "full transcript text",
  "channel_title": "Channel Name",
  "view_count": 12345,
  "duration_seconds": 360,
  "embedding": [0.001, 0.23, ...]       # required (list of floats) or stringified list
}

Run:
    uvicorn app:app --reload --host 0.0.0.0 --port 8000
"""
from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Any, Union
from pathlib import Path
import os
import ast
import json
import pandas as pd
from sentence_transformers import SentenceTransformer

# Optional: load .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

DB_PERSIST_DIR = os.getenv("DB_PERSIST_DIR", "chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "youtube_videos")
BATCH_SIZE = int(os.getenv("INGEST_BATCH_SIZE", "256"))
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "all-MiniLM-L6-v2")

app = FastAPI(title="Chroma Ingest API", version="1.0")

# Initialize embedder (cached)
_embedder = None
def get_embedder():
    global _embedder
    if _embedder is None:
        print(f"Loading embedding model: {EMBED_MODEL_NAME}")
        _embedder = SentenceTransformer(EMBED_MODEL_NAME)
    return _embedder

# ----------------------------
# Utility helpers
# ----------------------------
def ensure_embedding_list(x: Any, text_fallback: str = None):
    """
    Accept list, tuple, JSON-string, or Python-list string, return list[float] or None.
    If x is invalid or missing and text_fallback is provided, generate embedding from text.
    """
    if x is None or x == "":
        # If embedding is missing, try to generate from text
        if text_fallback:
            print(f"  Generating embedding from text (length={len(text_fallback)})")
            try:
                model = get_embedder()
                emb = model.encode(text_fallback, convert_to_numpy=True).tolist()
                return emb
            except Exception as e:
                print(f"  Warning: Failed to generate embedding: {e}")
                return None
        return None
    
    if isinstance(x, (list, tuple)):
        return list(x)
    if isinstance(x, str):
        s = x.strip()
        if s == "":
            if text_fallback:
                try:
                    model = get_embedder()
                    emb = model.encode(text_fallback, convert_to_numpy=True).tolist()
                    return emb
                except Exception:
                    return None
            return None
        # try JSON
        try:
            val = json.loads(s)
            if isinstance(val, list):
                return list(val)
        except Exception:
            pass
        # try ast literal_eval for python list syntax
        try:
            val = ast.literal_eval(s)
            if isinstance(val, (list, tuple)):
                return list(val)
        except Exception:
            pass
    
    # If all parsing failed, try to generate from text
    if text_fallback:
        try:
            model = get_embedder()
            emb = model.encode(text_fallback, convert_to_numpy=True).tolist()
            return emb
        except Exception:
            pass
    
    return None

def make_safe_id(raw_id: Optional[str], idx: int, used_ids: set):
    """
    Return a safe unique id string. If raw_id is missing or invalid, fallback to vid_{idx}.
    Ensure uniqueness by appending suffix if needed.
    """
    base = None
    if raw_id is not None:
        if isinstance(raw_id, (int, float)):
            base = str(raw_id)
        else:
            s = str(raw_id).strip()
            if s and s.lower() != "nan":
                base = s
    if not base:
        base = f"vid_{idx}"
    final = base
    counter = 1
    while final in used_ids:
        final = f"{base}_{counter}"
        counter += 1
    used_ids.add(final)
    return final

def create_chroma_client(persist_path: str):
    """Return a chroma client instance bound to persist_path (tries multiple constructors)."""
    try:
        import chromadb
    except Exception as e:
        raise RuntimeError("chromadb not installed. pip install chromadb") from e

    p = Path(persist_path)
    p.mkdir(parents=True, exist_ok=True)

    # Try PersistentClient
    try:
        if hasattr(chromadb, "PersistentClient"):
            client = chromadb.PersistentClient(path=str(p))
            return client
    except Exception:
        pass

    # Try Settings
    try:
        from chromadb.config import Settings
        client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=str(p)))
        return client
    except Exception:
        pass

    # Try path arg
    try:
        client = chromadb.Client(path=str(p))
        return client
    except Exception:
        pass

    # fallback
    try:
        client = chromadb.Client()
        return client
    except Exception as e:
        raise RuntimeError("Failed to create chromadb client: " + str(e)) from e

def get_or_create_collection(client, name: str):
    """Return collection object using best available API."""
    if hasattr(client, "get_or_create_collection"):
        return client.get_or_create_collection(name=name)
    if hasattr(client, "get_collection"):
        try:
            return client.get_collection(name=name)
        except Exception:
            # if not exists, try create_collection if available
            if hasattr(client, "create_collection"):
                return client.create_collection(name=name)
            raise
    if hasattr(client, "create_collection"):
        return client.create_collection(name=name)
    raise RuntimeError("Unable to create/get collection on client.")

# ----------------------------
# Pydantic models
# ----------------------------
class IngestRecord(BaseModel):
    video_id: Optional[str] = Field(None, description="Video ID (optional). Will be generated if missing.")
    title: Optional[str] = None
    transcript: Optional[str] = None
    channel_title: Optional[str] = None
    view_count: Optional[int] = None
    duration_seconds: Optional[int] = None
    embedding: Any = Field(..., description="Embedding vector as list or stringified list")

# ----------------------------
# Ingest helper (core logic)
# ----------------------------
def upsert_records_into_chroma(records: List[dict], persist_dir: str = DB_PERSIST_DIR, collection_name: str = COLLECTION_NAME):
    """
    Convert records (list of dict) to ids, documents, metadatas, embeddings and upsert to Chroma collection.
    Returns dict summary.
    """
    client = create_chroma_client(persist_dir)
    collection = get_or_create_collection(client, collection_name)

    ids = []
    documents = []
    metadatas = []
    embeddings = []

    used_ids = set()
    skipped = 0
    for idx, rec in enumerate(records):
        raw_id = rec.get("video_id") or rec.get("id")
        vid = make_safe_id(raw_id, idx, used_ids)

        # pick doc text first (needed for auto-embedding)
        doc = rec.get("transcript") or rec.get("transcript_clean") or rec.get("combined_text") or ""
        if pd.isna(doc) if isinstance(doc, float) else False:
            doc = ""
        doc = str(doc) if doc is not None else ""

        # Get or generate embedding (pass doc for fallback generation)
        emb = ensure_embedding_list(rec.get("embedding"), text_fallback=doc)
        if emb is None:
            skipped += 1
            print(f"Row {idx} (video_id={vid}): No valid embedding and no transcript to generate from. Skipping.")
            continue

        ids.append(vid)
        documents.append(doc)
        metadatas.append({
            "title": str(rec.get("title") or rec.get("title_clean") or ""),
            "channel_title": str(rec.get("channel_title") or ""),
            "view_count": int(rec.get("view_count")) if rec.get("view_count") not in (None, "") and str(rec.get("view_count")).isdigit() else None,
            "duration_seconds": int(rec.get("duration_seconds")) if rec.get("duration_seconds") not in (None, "") else None
        })
        embeddings.append(emb)

    if not ids:
        return {"inserted": 0, "skipped": skipped, "total": len(records)}

    # Upsert in batches
    B = BATCH_SIZE
    inserted = 0
    for i in range(0, len(ids), B):
        j = min(i + B, len(ids))
        batch_ids = ids[i:j]
        batch_docs = documents[i:j]
        batch_meta = metadatas[i:j]
        batch_emb = embeddings[i:j]
        # Try common add signature
        try:
            collection.add(ids=batch_ids, documents=batch_docs, metadatas=batch_meta, embeddings=batch_emb)
        except TypeError:
            # try positional
            try:
                collection.add(batch_ids, batch_docs, batch_meta, batch_emb)
            except Exception:
                if hasattr(collection, "upsert"):
                    collection.upsert(ids=batch_ids, documents=batch_docs, metadatas=batch_meta, embeddings=batch_emb)
                else:
                    raise
        inserted += (j - i)

    # persist if available
    try:
        if hasattr(client, "persist"):
            client.persist()
    except Exception:
        pass

    return {"inserted": inserted, "skipped": skipped, "total": len(records)}

# ----------------------------
# API endpoints
# ----------------------------

@app.get("/")
def root():
    """Root endpoint - redirect to API docs."""
    return RedirectResponse(url="/docs")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/ingest/json")
async def ingest_json(payload: Union[dict, list] = Body(..., description="Single record or list of records")):
    """
    Ingest JSON payload. Payload can be:
      - a single object (dict)
      - a list of objects
    Each object must have 'embedding' field (list or stringified list).
    """
    try:
        # normalize to list
        if isinstance(payload, dict):
            records = [payload]
        elif isinstance(payload, list):
            records = payload
        else:
            raise HTTPException(status_code=400, detail="Payload must be a JSON object or array of objects.")
        summary = upsert_records_into_chroma(records, persist_dir=DB_PERSIST_DIR, collection_name=COLLECTION_NAME)
        return JSONResponse({"status": "success", "detail": summary})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest/csv")
async def ingest_csv(file: UploadFile = File(...)):
    """
    Upload a CSV file. CSV must contain an 'embedding' column (stringified list or JSON list).
    Other columns mapped directly to metadata/document fields: video_id, title, transcript, channel_title, view_count, duration_seconds
    """
    try:
        contents = await file.read()
        # read into pandas
        df = pd.read_csv(pd.io.common.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read CSV: {e}")

    # Convert rows to records
    records = []
    for _, row in df.iterrows():
        rec = {}
        for c in df.columns:
            rec[c] = row[c]
        records.append(rec)

    try:
        summary = upsert_records_into_chroma(records, persist_dir=DB_PERSIST_DIR, collection_name=COLLECTION_NAME)
        return JSONResponse({"status": "success", "detail": summary})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
