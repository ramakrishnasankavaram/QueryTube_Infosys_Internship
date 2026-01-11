import ast
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

INPUT_CSV = "C:/Users/ramak/OneDrive/Desktop/InfosysSpringBoard/new_folder/master_data_preprocessed.csv"
EMBED_CSV = "master_data_with_embeddings.csv"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"  # small & fast, good default

def embed_texts(texts, model_name=EMBED_MODEL_NAME, batch_size=64):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)
    return embeddings

def main():
    p = Path(INPUT_CSV)
    if not p.exists():
        raise FileNotFoundError(f"Preprocessed CSV not found at {INPUT_CSV}. Run preprocess.py first.")
    df = pd.read_csv(p)
    if "combined_text" not in df.columns:
        raise ValueError("combined_text column not found. Make sure you ran preprocess.py")

    texts = df["combined_text"].fillna("").tolist()
    print(f"Embedding {len(texts)} items using {EMBED_MODEL_NAME} ...")
    embeddings = embed_texts(texts)

    # store embedding as list in csv (stringified)
    df["embedding"] = [emb.tolist() for emb in embeddings]
    df.to_csv(EMBED_CSV, index=False)
    print(f"Saved embeddings to {EMBED_CSV}")

if __name__ == '__main__':
    main()