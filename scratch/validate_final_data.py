
import pandas as pd
import numpy as np

def validate_results():
    features_path = r'd:\trading_project\data\processed\features\features.parquet'
    targets_path = r'd:\trading_project\data\processed\features\targets.parquet'
    
    print(f"--- Loading data ---")
    features = pd.read_parquet(features_path)
    targets = pd.read_parquet(targets_path)
    
    print(f"Features shape: {features.shape}")
    print(f"Targets shape: {targets.shape}")
    
    # Check daily targets
    daily_targets = [c for c in targets.columns if '_1d' in c]
    print(f"\nDaily targets found: {daily_targets}")
    
    if len(daily_targets) >= 2:
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        corr = targets[daily_targets].corr()
        print("\nCorrelation matrix for daily targets:")
        print(corr)
        
        # Check if they are 100% correlated
        for i in range(len(daily_targets)):
            for j in range(i + 1, len(daily_targets)):
                c = corr.iloc[i, j]
                if c > 0.99:
                    print(f"WARNING: {daily_targets[i]} and {daily_targets[j]} are highly correlated ({c:.4f})")
                else:
                    print(f"SUCCESS: {daily_targets[i]} and {daily_targets[j]} are distinct ({c:.4f})")

    # Check for NaNs
    nan_counts = targets.isna().sum()
    print("\nNaN counts in targets:")
    print(nan_counts[nan_counts > 0])
    
    # Distribution of some targets
    print("\nTarget distributions (describe):")
    print(targets[daily_targets].describe())

if __name__ == "__main__":
    validate_results()
