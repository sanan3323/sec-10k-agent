import numpy as np
from sqlalchemy import create_engine, text
from fastembed import TextEmbedding 

# Connect to your local Docker DB
DB_URL = "postgresql+psycopg://postgres:postgres@localhost:5432/sec10k"
engine = create_engine(DB_URL)

print("🚀 Loading Intel-optimized embedding model...")
# This uses ONNX (no PyTorch needed!)
model = TextEmbedding("BAAI/bge-large-en-v1.5")

queries = [
    ("AAPL supply chain risks",     {"ticker": "AAPL", "section": "Item 1A"}),
    ("JPMorgan credit risk",        {"ticker": "JPM",  "section": "Item 1A"}),
]

for query, filters in queries:
    print("\n" + "=" * 70)
    print(f"Query: {query}  filters={filters}")
    print("=" * 70)

    # FastEmbed uses a slightly different call than SentenceTransformer
    vec = list(model.embed([query]))[0]
    vec_literal = "[" + ",".join(repr(float(x)) for x in vec) + "]"

    sql = """
        SELECT 
          ticker, fiscal_year, section,
          LEFT(text, 180) AS preview,
          embedding <=> CAST(:q AS vector) AS distance
        FROM text_chunks
        WHERE ticker = :ticker AND section = :section
        ORDER BY distance ASC
        LIMIT 3
    """
    
    with engine.connect() as conn:
        rows = conn.execute(text(sql), {"q": vec_literal, **filters}).fetchall()

    if not rows:
        print("❌ No matches found. Did you load the data into the DB yet?")
    
    for i, row in enumerate(rows, 1):
        print(f"\n  [{i}] {row.ticker} FY{row.fiscal_year} {row.section}  d={row.distance:.4f}")
        print(f"      {row.preview}...")
