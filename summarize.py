#!/usr/bin/env python3
"""
summarize.py - Interactive version
----------------------------------
- Fetches transcript from Chroma by video_id (user input)
- Summarizes using Google Gemini (or local fallback)
- Uses .env for configuration
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configurable variables
DB_PERSIST_DIR = os.getenv("DB_PERSIST_DIR", "chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "youtube_videos")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
CHUNK_SIZE = int(os.getenv("SUMMARY_CHUNK_SIZE", "4000"))
CHUNK_OVERLAP = int(os.getenv("SUMMARY_CHUNK_OVERLAP", "200"))

# --------------------- Chroma Helpers ---------------------
def create_client_for_path(persist_path: Path):
    """Try several constructors to open the Chroma DB at persist_path."""
    try:
        import chromadb
    except Exception:
        raise RuntimeError("Please install chromadb: pip install chromadb")

    # PersistentClient (preferred)
    try:
        if hasattr(chromadb, "PersistentClient"):
            client = chromadb.PersistentClient(path=str(persist_path))
            print(f"[client] Connected via PersistentClient(path={persist_path})")
            return client
    except Exception as e:
        print("[client] PersistentClient failed:", e)

    # Settings-based constructor
    try:
        from chromadb.config import Settings
        client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=str(persist_path)))
        print(f"[client] Connected via Client(Settings(persist_directory={persist_path}))")
        return client
    except Exception as e:
        print("[client] Settings constructor failed:", e)

    # Path-based constructor
    try:
        client = chromadb.Client(path=str(persist_path))
        print(f"[client] Connected via Client(path={persist_path})")
        return client
    except Exception as e:
        print("[client] Client(path=...) failed:", e)

    # Fallback in-memory
    client = chromadb.Client()
    print("[client] Using in-memory Chroma (no persistence)")
    return client


def get_collection(client, name):
    """Get or create the collection."""
    try:
        if hasattr(client, "get_collection"):
            return client.get_collection(name=name)
    except Exception:
        pass
    try:
        if hasattr(client, "get_or_create_collection"):
            return client.get_or_create_collection(name=name)
    except Exception:
        pass
    raise RuntimeError(f"Unable to get collection '{name}'")


def fetch_document(collection, video_id):
    """Fetch transcript for a given video_id."""
    # Try .get()
    try:
        if hasattr(collection, "get"):
            res = collection.get(ids=[video_id], include=["documents"])
            docs = res.get("documents", [])
            if docs:
                first = docs[0]
                if isinstance(first, list):
                    return first[0] if first else ""
                return first
    except Exception:
        pass

    # Try .query()
    try:
        if hasattr(collection, "query"):
            res = collection.query(ids=[video_id], include=["documents"], n_results=1)
            docs = res.get("documents", [])
            if docs:
                first = docs[0]
                if isinstance(first, list):
                    return first[0] if first else ""
                return first
    except Exception:
        pass
    return None

# --------------------- Summarization ---------------------
def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into overlapping chunks."""
    if not text:
        return []
    text = text.strip()
    if len(text) <= size:
        return [text]
    chunks, start = [], 0
    while start < len(text):
        end = min(len(text), start + size)
        chunks.append(text[start:end])
        start = max(end - overlap, end)
    return chunks


def summarize_with_gemini(text: str) -> str:
    """Summarize using Google Gemini API."""
    try:
        from google import genai
    except Exception:
        raise RuntimeError("Install Google GenAI SDK: pip install google-genai")

    if not GEMINI_API_KEY:
        raise RuntimeError("Set GEMINI_API_KEY in .env")

    client = genai.Client(api_key=GEMINI_API_KEY)
    prompt = (
        "You are an expert summarizer. Produce a concise and effective summary (6 sentences) capturing the key ideas "
        "and main insights.\n\n"
        f"Transcript:\n{text}\n\nSummary:"
    )

    response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
    if hasattr(response, "text"):
        return response.text.strip()
    return str(response).strip()


def summarize_with_transformers(text: str) -> str:
    """Fallback summarizer using HuggingFace transformers."""
    try:
        from transformers import pipeline
    except Exception:
        raise RuntimeError("Install transformers: pip install transformers")

    summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=-1)
    out = summarizer(text[:1024], max_length=200, min_length=40, do_sample=False)
    return out[0]["summary_text"].strip()


def two_stage_summary(text: str) -> str:
    """Chunk large text and summarize in two stages."""
    chunks = chunk_text(text)
    summaries = []
    for i, chunk in enumerate(chunks, 1):
        print(f"[info] Summarizing chunk {i}/{len(chunks)} ({len(chunk)} chars)")
        try:
            summaries.append(summarize_with_gemini(chunk))
        except Exception as e:
            print("[warn] Gemini failed, using local summarizer:", e)
            summaries.append(summarize_with_transformers(chunk))

    combined = "\n\n".join(summaries)
    try:
        return summarize_with_gemini(combined)
    except Exception:
        return summarize_with_transformers(combined)


# --------------------- Main ---------------------
def main():
    print("=" * 60)
    print("🎥  YouTube Video Summarizer (Chroma + Gemini)")
    print("=" * 60)

    # Ask for video ID
    video_id = input("\nEnter video ID: ").strip()
    if not video_id:
        print("❌ No video ID entered.")
        return

    persist_path = Path(DB_PERSIST_DIR).expanduser().resolve()
    print(f"\n[info] Using DB path: {persist_path}")
    client = create_client_for_path(persist_path)
    collection = get_collection(client, COLLECTION_NAME)

    # Fetch transcript
    transcript = fetch_document(collection, video_id)
    if not transcript:
        print(f"❌ No transcript found for video ID '{video_id}' in '{COLLECTION_NAME}'.")
        return

    print(f"\n✅ Transcript found ({len(transcript)} characters).")
    print("\n--- Transcript Preview ---")
    print(transcript[:500])
    print("\n---------------------------")

    # Summarize
    print("\n⏳ Generating summary...")
    try:
        if len(transcript) <= CHUNK_SIZE:
            if GEMINI_API_KEY:
                summary = summarize_with_gemini(transcript)
            else:
                summary = summarize_with_transformers(transcript)
        else:
            summary = two_stage_summary(transcript)
    except Exception as e:
        print("[error] Summarization failed:", e)
        print("[info] Trying fallback summarizer...")
        summary = summarize_with_transformers(transcript)

    print("\n" + "=" * 60)
    print(f"📘 Summary for video ID: {video_id}\n")
    print(summary)
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
