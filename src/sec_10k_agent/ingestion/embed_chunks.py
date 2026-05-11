from pathlib import Path

import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sqlalchemy import create_engine

""" Using in google colab"""

DB_URL = "postgresql://postgres:postgres@localhost:5432/sec10k"

def generate_embeddings():
    chunks_path = Path("data/processed/chunks.parquet")
    if not chunks_path.exists():
        print("Chunks file not found!")
        return
    
    df = pd.read_parquet(chunks_path)
    print(f"Loaded {len(df)} chunks.")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Loading BGE model on {device}...")
    model = SentenceTransformer('BAAI/bge-large-en-v1.5', device=device)

    print("Generating vectors (this might take a minute)...")
    embeddings = model.encode(df['content'].tolist(), show_progress_bar=True)
    
    df['embedding'] = embeddings.tolist()
    
    engine = create_engine(DB_URL)
    print("Uploading to 'text_chunks'")
    
    if 'id' in df.columns:
        df = df.drop(columns=['id'])
        
    df.to_sql('text_chunks', engine, if_exists='append', index=False, method='multi', chunksize=200)
    print("Done")

if __name__ == "__main__":
    generate_embeddings()


"""GOOGLE COLAB VERSION

Because of hardware issues I ran the code in Colab


import pandas as pd
from sentence_transformers import SentenceTransformer
import torch
from pathlib import Path


base_path = Path("/content/drive/Othercomputers/Current MBP/sec-10k-agent")
input_file = base_path / "data" / "processed" / "chunks.parquet"
output_file = base_path / "data" / "processed" / "chunks_with_vectors.parquet"


if not input_file.exists():
    print(f"{input_file} not found.")
else:
    df = pd.read_parquet(input_file)

    
   
    target_col = None
    for col in ['content', 'text', 'chunk']:
        if col in df.columns:
            target_col = col
            break

    if not target_col:
        target_col = df.columns[0] # Fallback to the first column if names don't match
        print(f"Neither 'content' nor 'text' found. Using column: '{target_col}'")
    else:
        print(f"Found data in column: '{target_col}'")

    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer('BAAI/bge-large-en-v1.5', device=device)

    print(f"Vectorizing {len(df)} rows on {device}")
    embeddings = model.encode(df[target_col].tolist(), show_progress_bar=True)

    df['embedding'] = embeddings.tolist()
    df.to_parquet(output_file)
    print(f"Saved to {output_file}") """
