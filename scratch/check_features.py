import pandas as pd
from pathlib import Path

def main():
    feat_path = Path("d:/trading_project/cache/macro_data.parquet")
    if not feat_path.exists():
        print("macro_data.parquet not found!")
        return
        
    df = pd.read_parquet(feat_path)
    print(f"Shape: {df.shape}")
    print("Columns in cached macro_data:")
    for col in df.columns:
        print(f"  {col}")

if __name__ == "__main__":
    main()
