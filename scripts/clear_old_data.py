#!/usr/bin/env python3
"""
Script to clear old data from the database and start fresh with 60-day limit.
"""

import os
import sys
import shutil
from pathlib import Path

def clear_old_data():
    """Clear old accumulated data to start fresh with 60-day limit."""
    
    # Paths to clear
    data_paths = [
        "data/colab/accumulated/main_database",
        "data/colab/accumulated/processed",
        "data/colab/accumulated/raw",
        "data/colab/accumulated/temp",
    ]
    
    print("Clearing old data to start fresh with 60-day limit...")
    
    for path_str in data_paths:
        path = Path(path_str)
        if path.exists():
            print(f"  Removing: {path}")
            if path.is_file():
                path.unlink()
            else:
                shutil.rmtree(path)
        else:
            print(f"  Not found: {path}")
    
    # Also clear cache
    cache_path = Path("data/cache")
    if cache_path.exists():
        print(f"  Clearing cache: {cache_path}")
        shutil.rmtree(cache_path)
    
    print("\n✅ Old data cleared. Ready to run pipeline with 60-day limit.")
    print("\nNext steps:")
    print("1. Run: python run_hybrid_pipeline.py --mode prepare")
    print("2. Verify data spans only 60 days")
    print("3. Check 15m data has sufficient rows")

if __name__ == "__main__":
    clear_old_data()