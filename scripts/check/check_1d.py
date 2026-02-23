#!/usr/bin/env python3
"""
Перевandрка структури 1d fileу whereandльно
"""

import os
import pandas as pd

project_root = os.path.dirname(os.path.abspath(__file__))

price_file = os.path.join(project_root, "data", "stages", "prices_1d.parquet")
if os.path.exists(price_file):
    print("=== 1d DETAILED ===")
    df = pd.read_parquet(price_file)
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    if len(df) > 0:
        print(f"Sample:\n{df.head(10)}")
        
        # Перевandряємо чи є multi-index
        if hasattr(df.columns, 'levels'):
            print(f"Column levels: {df.columns.levels}")
            
        # Перевandряємо першand рядки
        print(f"First row keys: {df.iloc[0].to_dict() if len(df) > 0 else 'Empty'}")
