import re
import ast
import pandas as pd
from pathlib import Path

CSV_PATH = "C:/Users/ramak/OneDrive/Desktop/InfosysSpringBoard/master_dataset_updated.csv"  # update path if needed
OUTPUT_CSV = "master_data_preprocessed.csv"

def parse_duration_to_seconds(dur_str):
    """
    Convert YouTube-like ISO 8601 duration strings like 'PT1H2M30S', 'PT20M30S', 'PT45S'
    or numeric seconds to integer seconds. Returns None for missing/invalid.
    """
    if pd.isna(dur_str):
        return None
    if isinstance(dur_str, (int, float)):
        return int(dur_str)
    s = str(dur_str).strip().upper()
    # If it's already digits (or digits in string), return int
    if s.isdigit():
        return int(s)
    # regex extract hours/minutes/seconds
    match = re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", s)
    if match:
        h = int(match.group(1) or 0)
        m = int(match.group(2) or 0)
        sec = int(match.group(3) or 0)
        return h*3600 + m*60 + sec
    # try fallback parsing like '20m30s' or '1:20:30'
    try:
        # colon format HH:MM:SS or MM:SS
        if ":" in s:
            parts = [int(p) for p in s.split(":")]
            if len(parts) == 3:
                return parts[0]*3600 + parts[1]*60 + parts[2]
            elif len(parts) == 2:
                return parts[0]*60 + parts[1]
        # fallback: find digits with units
        found = re.findall(r"(\d+)\s*h", s)
        if found:
            h = int(found[0])
        else:
            h = 0
        found = re.findall(r"(\d+)\s*m", s)
        m = int(found[0]) if found else 0
        found = re.findall(r"(\d+)\s*s", s)
        sec = int(found[0]) if found else 0
        total = h*3600 + m*60 + sec
        if total>0:
            return total
    except Exception:
        pass
    return None

def clean_text(text):
    """Lowercase, replace multiple whitespace with single space, remove control chars, and strip most special characters but keep basic punctuation."""
    if pd.isna(text):
        return ""
    if isinstance(text, (list, dict)):
        text = str(text)
    text = str(text)
    # convert bytes-like to str safely
    text = text.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")
    text = text.lower()
    # remove non-printable/control characters
    text = re.sub(r"[\r\n\t]+", " ", text)
    # replace unusual unicode dash/quotes with ascii equivalents
    text = text.replace("–", "-").replace("—", "-").replace("“", "\"").replace("”", "\"").replace("’", "'")
    # remove any characters except letters, numbers, basic punctuation and whitespace
    text = re.sub(r"[^0-9a-zA-Z\s\.\,\!\?\:\'\-\"/()]", " ", text)
    # collapse multiple spaces
    text = re.sub(r"\s{2,}", " ", text).strip()
    return text

def main():
    p = Path(CSV_PATH)
    if not p.exists():
        raise FileNotFoundError(f"CSV not found at {CSV_PATH}. Place your dataset in the working dir or update CSV_PATH.")
    df = pd.read_csv(p)
    print(f"Loaded {len(df)} rows from {CSV_PATH}")

    # standardize columns existence
    for col in ["title", "transcript", "duration"]:
        if col not in df.columns:
            df[col] = ""

    # clean and lowercase title and transcript
    df["title_clean"] = df["title"].apply(clean_text)
    df["transcript_clean"] = df["transcript"].apply(clean_text)

    # convert duration to seconds
    df["duration_seconds"] = df["duration"].apply(parse_duration_to_seconds)

    # combined text for embedding
    df["combined_text"] = (df["title_clean"].fillna("") + " " + df["transcript_clean"].fillna("")).str.strip()

    # save
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved preprocessed data to {OUTPUT_CSV}")

if __name__ == '__main__':
    main()