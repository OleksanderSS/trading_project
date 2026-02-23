#!/usr/bin/env python3
import pandas as pd

print("=== CHECKING 15m DATA AND DATE ISSUES ===")

df = pd.read_parquet('c:/trading_project/data/stages/merged_full.parquet')

# Check 15m data specifically
print(f"Dataset shape: {df.shape}")

# Check 15m close prices
close_15m_cols = [col for col in df.columns if '15m' in col and 'close' in col]
print(f"\n15m close columns: {close_15m_cols}")

if close_15m_cols:
    for col in close_15m_cols[:3]:  # Check first 3
        non_null = df[col].notna().sum()
        unique_vals = df[col].nunique()
        print(f"  {col}: {non_null}/{len(df)} non-null, {unique_vals} unique values")
        print(f"    Sample values: {df[col].dropna().head(5).tolist()}")

# Check 60m data for comparison
close_60m_cols = [col for col in df.columns if '60m' in col and 'close' in col]
print(f"\n60m close columns: {close_60m_cols}")

if close_60m_cols:
    for col in close_60m_cols[:3]:
        non_null = df[col].notna().sum()
        unique_vals = df[col].nunique()
        print(f"  {col}: {non_null}/{len(df)} non-null, {unique_vals} unique values")
        print(f"    Sample values: {df[col].dropna().head(5).tolist()}")

# Check date formats
print(f"\nDate columns:")
date_cols = ['published_at', 'event_time', 'event_time_for_intraday', 'trade_date']
for col in date_cols:
    if col in df.columns:
        print(f"  {col}: {df[col].dtype}")
        print(f"    Sample: {df[col].dropna().head(3).tolist()}")

# Check for timestamp format issues
print(f"\nChecking for large timestamp numbers:")
for col in date_cols:
    if col in df.columns:
        max_val = df[col].max()
        min_val = df[col].min()
        print(f"  {col}: min={min_val}, max={max_val}")
