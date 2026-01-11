"""
vector_db.py (fixed)
- Loads EMBED_CSV and upserts selected fields into a Chroma vector DB (local persistence)
- Ensures documents are strings (empty string when missing) to avoid "Expected document to be a str, got nan" errors.
- Fixes duplicate ID issue by ensuring each video has a unique valid ID.
- Creates persistent ChromaDB directory properly.
- Tries to be compatible with multiple chromadb versions.
"""

import ast
import json
import pandas as pd
from pathlib import Path

CSV_WITH_EMB = "master_data_with_embeddings.csv"
PERSIST_DIR = "chroma_db"
COLLECTION_NAME = "youtube_videos"

def ensure_embedding_list(x):
    if pd.isna(x) or x is None or x == "":
        return None
    if isinstance(x, str):
        try:
            val = ast.literal_eval(x)
            return list(val)
        except Exception:
            try:
                val = json.loads(x)
                return list(val)
            except Exception:
                return None
    if isinstance(x, (list, tuple)):
        return list(x)
    return None

def create_chroma_client(persist_directory):
    import chromadb
    
    # Create directory if it doesn't exist
    persist_path = Path(persist_directory)
    persist_path.mkdir(exist_ok=True)
    
    # Try PersistentClient first (newer versions)
    try:
        client = chromadb.PersistentClient(path=str(persist_path))
        print(f"Chroma client created with chromadb.PersistentClient(path='{persist_path}')")
        return client
    except AttributeError:
        print("PersistentClient not available, trying other methods...")
    except Exception as ex:
        print(f"chromadb.PersistentClient() failed: {ex}")
    
    # Try HttpClient for server-based setup
    try:
        client = chromadb.HttpClient()
        print("Chroma client created with chromadb.HttpClient()")
        return client
    except Exception:
        pass

    # Try Settings approach (older versions)
    try:
        from chromadb.config import Settings
        client = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=str(persist_path)
        ))
        print(f"Chroma client created with Settings(persist_directory='{persist_path}')")
        return client
    except Exception as ex:
        print(f"Settings(...) approach failed: {ex}")

    # Try simple Client with path
    try:
        client = chromadb.Client(path=str(persist_path))
        print(f"Chroma client created with chromadb.Client(path='{persist_path}')")
        return client
    except Exception as ex:
        print(f"chromadb.Client(path=...) failed: {ex}")

    raise RuntimeError(
        "Failed to create a chromadb client. Try upgrading chromadb (`pip install --upgrade chromadb`) "
        "or install chroma-migrate if migrating old data (`pip install chroma-migrate`).\n"
        "Current chromadb version should be 0.4.0 or higher for PersistentClient support."
    )

def get_or_create_collection(client, name):
    try:
        if hasattr(client, "get_or_create_collection"):
            collection = client.get_or_create_collection(name=name)
            print(f"Using client.get_or_create_collection() for collection '{name}'")
            return collection
        if hasattr(client, "create_collection"):
            try:
                collection = client.create_collection(name=name)
                print(f"Using client.create_collection() for collection '{name}'")
                return collection
            except Exception:
                if hasattr(client, "get_collection"):
                    collection = client.get_collection(name=name)
                    print(f"Using client.get_collection() for collection '{name}' (already exists)")
                    return collection
        if hasattr(client, "get_collection"):
            collection = client.get_collection(name=name)
            print(f"Using client.get_collection() for collection '{name}'")
            return collection
    except Exception as e:
        print(f"Collection creation/getting error: {e}")

    raise RuntimeError("Unable to create or fetch collection from chromadb client.")

def get_valid_video_id(row, idx):
    """Extract a valid video ID or generate one if missing."""
    # Try multiple possible ID fields
    vid = row.get("video_id")
    if pd.notna(vid) and vid is not None and str(vid).strip() and str(vid).lower() != "nan":
        return str(vid).strip()
    
    vid = row.get("id")
    if pd.notna(vid) and vid is not None and str(vid).strip() and str(vid).lower() != "nan":
        return str(vid).strip()
    
    # Try URL-based extraction
    url = row.get("url")
    if pd.notna(url) and url is not None and "v=" in str(url):
        try:
            vid_from_url = str(url).split("v=")[1].split("&")[0]
            if vid_from_url:
                return vid_from_url
        except Exception:
            pass
    
    # Fallback: generate ID from index
    return f"vid_{idx}"

def main():
    p = Path(CSV_WITH_EMB)
    if not p.exists():
        raise FileNotFoundError(f"{CSV_WITH_EMB} not found. Run embed.py first.")
    
    print(f"Loading data from {CSV_WITH_EMB}...")
    df = pd.read_csv(p)
    print(f"Loaded {len(df)} rows")

    client = create_chroma_client(PERSIST_DIR)
    collection = get_or_create_collection(client, COLLECTION_NAME)

    ids = []
    metadatas = []
    embeddings = []
    documents = []
    
    # Track used IDs to ensure uniqueness
    used_ids = set()

    # iterate rows and build safe values
    for idx, row in df.iterrows():
        # Get valid video ID
        vid = get_valid_video_id(row, idx)
        
        # Ensure ID is unique (handle potential duplicates)
        original_vid = vid
        counter = 1
        while vid in used_ids:
            vid = f"{original_vid}_{counter}"
            counter += 1
        used_ids.add(vid)

        emb = ensure_embedding_list(row.get("embedding"))
        if emb is None:
            # skip rows with no embeddings
            print(f"Skipping row {idx} (video_id={vid}) because embedding is missing/invalid.")
            continue

        # Prefer transcript_clean, then combined_text, then transcript; ensure it's a string
        raw_doc = row.get("transcript_clean") if "transcript_clean" in row else None
        if not raw_doc or pd.isna(raw_doc):
            raw_doc = row.get("combined_text") if "combined_text" in row else None
        if not raw_doc or pd.isna(raw_doc):
            raw_doc = row.get("transcript") if "transcript" in row else None

        # Convert missing/nan to empty string and ensure str type
        if pd.isna(raw_doc) or raw_doc is None:
            safe_doc = ""
        else:
            safe_doc = str(raw_doc)

        # metadata
        try:
            viewc = row.get("view_count")
            viewc_int = int(viewc) if (viewc is not None and str(viewc).strip().isdigit()) else None
        except Exception:
            viewc_int = None

        duration_sec = None
        if "duration_seconds" in row and not pd.isna(row.get("duration_seconds")):
            try:
                duration_sec = int(row.get("duration_seconds"))
            except Exception:
                duration_sec = None

        ids.append(vid)
        documents.append(safe_doc)
        metadatas.append({
            "title": str(row.get("title") or row.get("title_clean") or ""),
            "channel_title": str(row.get("channel_title") or "") if "channel_title" in row else "",
            "view_count": viewc_int,
            "duration_seconds": duration_sec
        })
        embeddings.append(emb)

    if not ids:
        print("No valid rows to insert (no embeddings found). Exiting.")
        return

    # final sanity check lengths
    n = len(ids)
    assert len(documents) == n == len(metadatas) == len(embeddings), "Lengths of ids/documents/metadata/embeddings must match!"
    
    print(f"\nPreparing to insert {n} documents into ChromaDB collection '{COLLECTION_NAME}'...")

    BATCH = 256
    for i in range(0, n, BATCH):
        j = min(i + BATCH, n)
        batch_ids = ids[i:j]
        batch_docs = documents[i:j]
        batch_meta = metadatas[i:j]
        batch_emb = embeddings[i:j]
        try:
            collection.add(
                ids=batch_ids,
                documents=batch_docs,
                metadatas=batch_meta,
                embeddings=batch_emb
            )
            print(f"✓ Inserted batch {i//BATCH + 1}: rows {i} to {j-1}")
        except TypeError as te:
            print(f"collection.add(...) TypeError, trying positional args: {te}")
            try:
                collection.add(batch_ids, batch_docs, batch_meta, batch_emb)
                print(f"✓ Inserted batch {i//BATCH + 1}: rows {i} to {j-1} (positional)")
            except Exception as e2:
                print(f"Failed to add batch with positional args: {e2}")
                if hasattr(collection, "upsert"):
                    collection.upsert(ids=batch_ids, documents=batch_docs, metadatas=batch_meta, embeddings=batch_emb)
                    print(f"✓ Upserted batch {i//BATCH + 1}: rows {i} to {j-1}")
                else:
                    raise

    # persist (for older versions)
    try:
        if hasattr(client, "persist"):
            client.persist()
            print(f"\n✓ Client.persist() called")
    except Exception as e:
        print(f"Note: Error while persisting (may not be needed): {e}")

    print(f"\n{'='*60}")
    print(f"✓ SUCCESS: Inserted {n} documents into ChromaDB")
    print(f"✓ Collection: '{COLLECTION_NAME}'")
    print(f"✓ Persist directory: '{PERSIST_DIR}'")
    print(f"{'='*60}")
    
    # Verify the directory was created
    persist_path = Path(PERSIST_DIR)
    if persist_path.exists():
        print(f"\n✓ Directory '{PERSIST_DIR}' exists")
        files = list(persist_path.glob("*"))
        if files:
            print(f"✓ Contains {len(files)} files/folders:")
            for f in files[:5]:  # Show first 5
                print(f"  - {f.name}")
            if len(files) > 5:
                print(f"  ... and {len(files) - 5} more")
    else:
        print(f"\n⚠ Warning: Directory '{PERSIST_DIR}' was not created!")
        print("This may indicate an in-memory database. Check your chromadb version.")


if __name__ == '__main__':
    main()