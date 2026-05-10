import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from pgvector.sqlalchemy import Vector
from pathlib import Path

DB_URL = "postgresql+psycopg://postgres:postgres@localhost:5432/sec10k"

def load_to_postgres():
    engine = create_engine(DB_URL)
    file_path = Path("data/processed/chunks_with_vectors.parquet")
    
    if not file_path.exists():
        print(f"File not found at {file_path}")
        return

    df = pd.read_parquet(file_path)
    print(f"Loaded {len(df)} chunks.")

   
    if "content" in df.columns and "text" not in df.columns:
        df = df.rename(columns={"content": "text"})

        print("Converting vectors to Python lists...")
    df["embedding"] = df["embedding"].apply(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)

    print("Pushing to local Docker Postgres...")
    try:
       
        with engine.begin() as conn:
            conn.execute(text("TRUNCATE text_chunks;"))
            print("Table cleared...")

        df.to_sql(
            "text_chunks",
            engine,
            if_exists="append",
            index=False,
            method="multi",
            chunksize=100,
            dtype={"embedding": Vector(1024)} 
        )
        
        with engine.connect() as conn:
            count = conn.execute(text("SELECT COUNT(*) FROM text_chunks")).scalar()
            print(f"{count} rows are now live in your Vector DB.")

    except Exception as e:
        print(f"Failed: {e}")

if __name__ == "__main__":
    load_to_postgres()
