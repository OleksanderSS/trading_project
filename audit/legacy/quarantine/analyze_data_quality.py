import pandas as pd
import numpy as np

features = pd.read_parquet('data/colab/accumulated/main_database/features.parquet')
targets = pd.read_parquet('data/colab/accumulated/main_database/targets.parquet')

print("--- Quality Report ---")
print(f"Features - Total Rows: {len(features)}, NaNs: {features.isna().sum().sum()}")
print(f"Targets - Total Rows: {len(targets)}, NaNs: {targets.isna().sum().sum()}")

# Check for zero columns that might indicate failed processing
zero_cols = [col for col in features.columns if features[col].nunique() == 1 and features[col].iloc[0] == 0]
print(f"Columns with only zeros in features: {len(zero_cols)}")

# Check for Inf values
inf_count = np.isinf(features.select_dtypes(include=[np.number])).sum().sum()
print(f"Inf values in features: {inf_count}")

# Check targets for zero variance (unusable for training)
zero_var_targets = [col for col in targets.columns if targets[col].nunique() <= 1]
print(f"Targets with zero variance: {zero_var_targets}")
