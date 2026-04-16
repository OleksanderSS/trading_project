"""
Інспекція даних в batch_dir
"""
import pandas as pd
from pathlib import Path

batch_dir = Path("data/colab/accumulated/test_ticker_amd_target_return_1d_ep5_iter5")

print("=" * 80)
print("BATCH DATA INSPECTION")
print("=" * 80)

# Features
features_path = batch_dir / "features.parquet"
features_df = pd.read_parquet(features_path)

print(f"\nFEATURES: {features_df.shape}")
print(f"Columns: {features_df.columns.tolist()[:10]}...")
print(f"\nFirst 3 rows:")
print(features_df.head(3))
print(f"\nNaN count per column (top 10):")
print(features_df.isna().sum().sort_values(ascending=False).head(10))
print(f"\nTotal NaN rows: {features_df.isna().any(axis=1).sum()}")

# Targets
targets_path = batch_dir / "targets.parquet"
targets_df = pd.read_parquet(targets_path)

print(f"\n" + "=" * 80)
print(f"TARGETS: {targets_df.shape}")
print(f"Columns: {targets_df.columns.tolist()}")
print(f"\nFirst 5 rows:")
print(targets_df.head())
print(f"\nNaN count:")
print(targets_df.isna().sum())
print(f"\nTarget stats:")
print(targets_df['target_return_1d'].describe())

# Check alignment
print(f"\n" + "=" * 80)
print("ALIGNMENT CHECK:")
print(f"Features rows: {len(features_df)}")
print(f"Targets rows: {len(targets_df)}")
print(f"Features index: {features_df.index[:5].tolist()}")
print(f"Targets index: {targets_df.index[:5].tolist()}")
