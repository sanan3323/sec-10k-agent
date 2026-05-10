import pandas as pd
import json
import sqlalchemy
from sqlalchemy import create_engine
from pathlib import Path

DB_URL = "postgresql://postgres:postgres@localhost:5432/sec10k"

def clean_dimensions(dims):
    if not isinstance(dims, dict):
        return {}
    return {k: v for k, v in dims.items() if v is not None}

def load_data():
    engine = create_engine(DB_URL)
    xbrl_path = Path("data/processed/xbrl.parquet")
    
    if not xbrl_path.exists():
        print(f"File not found: {xbrl_path}")
        return

    df = pd.read_parquet(xbrl_path)
    
    if 'fiscal_year_focus' in df.columns:
        df = df.rename(columns={'fiscal_year_focus': 'fiscal_year'})

    df['dim_str'] = df['dimensions'].apply(lambda d: str(sorted(d.items())) if d else "")
    df = df.drop_duplicates(subset=['accession_number', 'concept', 'dim_str', 'period_end'])
    df = df.drop(columns=['dim_str'])

    df['dimensions'] = df['dimensions'].apply(clean_dimensions)
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    
    allowed_cols = [
        'cik', 'ticker', 'fiscal_year', 'concept', 'value', 
        'unit', 'period_start', 'period_end', 'dimensions', 'accession_number'
    ]
    df = df[[c for c in allowed_cols if c in df.columns]]

    print(f"Uploading {len(df)} unique rows")
    try:
        df.to_sql(
            'financial_facts', 
            engine, 
            if_exists='append', 
            index=False,
            dtype={'dimensions': sqlalchemy.types.JSON} 
        )
        print("Loaded")
    except Exception as e:
        print(f"Failed: {e}")

if __name__ == "__main__":
    load_data()
