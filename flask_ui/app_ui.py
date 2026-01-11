# app.py
import os
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import sqlite3
from pathlib import Path
import requests
from dotenv import load_dotenv
from utils import embed_text, query_top_k, create_chroma_client, get_collection, youtube_thumbnail_url, youtube_watch_url, summarize_text

load_dotenv()

APP_SECRET = os.getenv("FLASK_SECRET_KEY", "dev-secret")
FASTAPI_URL = os.getenv("FASTAPI_URL", "http://127.0.0.1:8000")
DB_PERSIST_DIR = os.getenv("DB_PERSIST_DIR", "chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "youtube_videos")
DEFAULT_CSV_PATH = os.getenv("DEFAULT_CSV_PATH", "/mnt/data/master_dataset_updated.csv")

app = Flask(__name__)
app.secret_key = APP_SECRET
UPLOAD_FOLDER = "uploads"
Path(UPLOAD_FOLDER).mkdir(exist_ok=True)

# ========== Simple SQLite for users ===========
DB_FILE = Path("querytube_users.db")
def init_db():
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL
    )
    """)
    conn.commit()
    conn.close()


# ---------- Helpers: metadata normalization ----------
def _safe_int_from(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        try:
            return int(value)
        except Exception:
            return None
    if isinstance(value, str):
        s = value.strip().replace(',', '').replace('+', '')
        # remove non-digit characters
        import re
        m = re.search(r"(\d+)", s)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                return None
        return None
    if isinstance(value, dict):
        # common nested shapes
        for k in ("viewCount", "views", "view_count"):
            if k in value:
                return _safe_int_from(value.get(k))
    return None


def _extract_channel(meta: dict):
    if not meta or not isinstance(meta, dict):
        return None
    # Candidate keys in order of preference
    keys = [
        'channel_title', 'channelTitle', 'channel', 'uploader', 'author', 'uploader_name', 'creator', 'publisher'
    ]
    for k in keys:
        v = meta.get(k)
        if v and isinstance(v, str) and v.strip():
            return v.strip()
    # fallback: maybe metadata contains nested dict with channel info
    for k in ('owner', 'ownerChannel', 'owner_name'):
        v = meta.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def create_user(username, password):
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    try:
        cur.execute("INSERT INTO users (username, password_hash) VALUES (?, ?)",
                    (username, generate_password_hash(password)))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def verify_user(username, password):
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute("SELECT password_hash FROM users WHERE username = ?", (username,))
    row = cur.fetchone()
    conn.close()
    if not row:
        return False
    return check_password_hash(row[0], password)

init_db()

# ========== Routes ===========
@app.route("/")
def home():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("index.html", user=session.get("user"))


@app.route("/how")
def how():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("how.html")


@app.route("/tips")
def tips():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("tips.html")

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form.get("username").strip()
        password = request.form.get("password").strip()
        if not username or not password:
            flash("Username and password required", "error")
            return redirect(url_for("signup"))
        ok = create_user(username, password)
        if not ok:
            flash("Username already exists", "error")
            return redirect(url_for("signup"))
        flash("Account created, please login", "success")
        return redirect(url_for("login"))
    return render_template("signup.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username").strip()
        password = request.form.get("password").strip()
        if verify_user(username, password):
            session["user"] = username
            flash("Logged in", "success")
            return redirect(url_for("home"))
        flash("Invalid credentials", "error")
        return redirect(url_for("login"))
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.pop("user", None)
    flash("Logged out", "info")
    return redirect(url_for("login"))

# Search endpoint (POST-form)
@app.route("/search", methods=["POST"])
def search():
    if "user" not in session:
        return redirect(url_for("login"))
    query = request.form.get("query", "").strip()
    if not query:
        flash("Please provide a query", "error")
        return redirect(url_for("home"))
    
    # Query top k
    print(f"[search] Query: {query}")
    try:
        results = query_top_k(query, k=10)  # Get more than 5 to ensure we have good results
        print(f"[search] Got {len(results)} results")
    except Exception as e:
        print(f"[search] Error: {e}")
        import traceback
        traceback.print_exc()
        flash(f"Search failed: {e}", "error")
        return redirect(url_for("home"))

    # results: list of (id, metadata, doc, similarity)
    if not results:
        flash("No results found for your query", "info")
        return redirect(url_for("home"))
    
    items = []
    for vid, meta, doc, similarity in results:
        try:
            # Skip empty/invalid results
            if not vid or not meta:
                continue
            
            # Extract title (required for display)
            title = (meta.get("title", "") or "").strip() if meta else ""
            if not title:
                title = f"Video {vid}"
            
            # Extract channel name robustly
            channel = _extract_channel(meta) or "Unknown"
            
            # Extract view count (will likely be None or missing from current dataset)
            view_count = None
            if meta and isinstance(meta, dict):
                for k in ("view_count", "viewCount", "views"):
                    if k in meta:
                        v = meta.get(k)
                        if v is not None and v != "" and str(v).lower() != "nan":
                            view_count = _safe_int_from(v)
                            if view_count is not None:
                                break
            
            # Extract duration
            duration = meta.get("duration_seconds") if meta else None
            
            # Build thumbnail and watch URLs
            thumb = youtube_thumbnail_url(vid)
            watch = youtube_watch_url(vid)
            
            # Format view count — show "Not available" if missing
            if isinstance(view_count, (int, float)) and view_count >= 0:
                if view_count >= 1000000:
                    formatted_views = f"{view_count/1000000:.1f}M"
                elif view_count >= 1000:
                    formatted_views = f"{view_count/1000:.1f}K"
                else:
                    formatted_views = str(int(view_count))
            else:
                formatted_views = "Not available"
            
            # Format duration (MM:SS)
            if duration and isinstance(duration, (int, float)):
                minutes = int(duration) // 60
                seconds = int(duration) % 60
                formatted_duration = f"{minutes}:{seconds:02d}"
            else:
                formatted_duration = "N/A"
            
            items.append({
                "video_id": vid,
                "title": title,
                "channel": channel,
                "view_count": formatted_views,
                "duration": formatted_duration,
                "thumbnail": thumb,
                "url": watch,
                "similarity": round(similarity * 100, 1),  # Convert to percentage
                "snippet": (doc[:250] + "...") if doc and len(doc) > 250 else (doc or "No description available")
            })
        except Exception as e:
            print(f"[search] Error processing result {vid}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Take only top 5 with highest similarity
    items = sorted(items, key=lambda x: x["similarity"], reverse=True)[:5]
    
    if not items:
        flash("Failed to process search results", "error")
        return redirect(url_for("home"))
    
    print(f"[search] Returning {len(items)} items to template")
    return render_template("results.html", items=items, query=query)

# Upload CSV (client -> Flask -> FastAPI /ingest/csv)
@app.route("/upload", methods=["GET", "POST"])
def upload():
    if "user" not in session:
        return redirect(url_for("login"))
    if request.method == "POST":
        f = request.files.get("csv_file")
        if not f or f.filename == "":
            flash("No file selected", "error")
            return redirect(url_for("upload"))
        
        # Validate file extension
        if not f.filename.lower().endswith('.csv'):
            flash("Please upload a CSV file", "error")
            return redirect(url_for("upload"))
        
        filename = secure_filename(f.filename)
        dst = Path(UPLOAD_FOLDER) / filename
        f.save(dst)
        
        # forward to FastAPI ingestion endpoint
        try:
            print(f"[upload] Uploading {filename} to FastAPI ingestion endpoint...")
            with open(dst, "rb") as file_obj:
                files = {"file": (filename, file_obj, "text/csv")}
                resp = requests.post(
                    f"{FASTAPI_URL.rstrip('/')}/ingest/csv",
                    files=files,
                    timeout=300  # increased timeout for large files
                )
            
            # Parse response
            try:
                data = resp.json()
                print(f"[upload] FastAPI response: {data}")
                
                # Extract useful info from response
                if resp.status_code == 200:
                    record_count = data.get("records_inserted", data.get("total_records", "?"))
                    ingest_msg = data.get("message", "File ingested successfully")
                    flash(
                        f"✅ Success! {ingest_msg} ({record_count} records processed)",
                        "success"
                    )
                else:
                    error_msg = data.get("detail", data.get("error", str(data)))
                    flash(f"⚠️ Ingest error: {error_msg}", "error")
            except Exception as json_err:
                # response is not JSON, show text
                print(f"[upload] Response not JSON: {json_err}")
                flash(f"✅ File uploaded. Server response: {resp.text[:200]}", "success")
        except requests.exceptions.Timeout:
            flash("⏱️ Upload timed out. The file may be too large or the server is slow.", "error")
        except requests.exceptions.ConnectionError:
            flash("🔌 Failed to connect to the ingestion server. Is the FastAPI server running?", "error")
        except Exception as e:
            print(f"[upload] Error: {e}")
            flash(f"❌ Upload failed: {str(e)[:150]}", "error")
        finally:
            # Clean up temp file
            try:
                dst.unlink()
            except Exception:
                pass
        
        return redirect(url_for("upload"))
    
    # GET
    return render_template("upload.html", default_csv_path=DEFAULT_CSV_PATH)

# Video summary page
@app.route("/summary/<video_id>")
def summary(video_id):
    if "user" not in session:
        return redirect(url_for("login"))
    # fetch document/transcript from Chroma collection directly
    try:
        client = create_chroma_client(DB_PERSIST_DIR)
        coll = get_collection(client, COLLECTION_NAME)
        # try collection.get(ids=[video_id])
        try:
            res = coll.get(ids=[video_id], include=["documents", "metadatas"])
        except Exception:
            res = coll.query(ids=[video_id], include=["documents", "metadatas"], n_results=1)
        docs = res.get("documents", [[]])[0] if isinstance(res.get("documents", []), list) else res.get("documents")
        transcript = docs[0] if docs and isinstance(docs, list) and docs[0] else (docs if isinstance(docs, str) else "")
    except Exception as e:
        flash(f"Failed to fetch transcript: {e}", "error")
        return redirect(url_for("home"))

    if not transcript:
        flash("No transcript found for this video", "error")
        return redirect(url_for("home"))

    # Summarize
    try:
        summary_text = summarize_text(transcript)
    except Exception as e:
        summary_text = f"Summarization failed: {e}"

    thumb = youtube_thumbnail_url(video_id)
    return render_template("summary.html", video_id=video_id, thumbnail=thumb, summary=summary_text, transcript_preview=transcript[:1000])

# Simple health check route (also accessible)
@app.route("/health")
def health():
    return jsonify({"status":"ok"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5500, debug=True)
