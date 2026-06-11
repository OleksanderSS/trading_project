
import pandas as pd
import json
import os

files = [
    "data/processed/features/enriched_features.parquet",
    "data/processed/features/news_features.parquet",
    "data/processed/features/targets.parquet"
]

results = {}

for f in files:
    path = os.path.join("d:/trading_project", f)
    if os.path.exists(path):
        df = pd.read_parquet(path)
        results[f] = {
            "shape": df.shape,
            "columns": df.columns.tolist()[:10], # Show first 10
            "tickers": df['ticker'].unique().tolist() if 'ticker' in df.columns else "N/A"
        }
    else:
        results[f] = {"status": "File not found"}

print(json.dumps(results, indent=2))
