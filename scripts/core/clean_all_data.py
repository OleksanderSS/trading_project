#!/usr/bin/env python3
"""
Clean all accumulated data and cache for fresh start.
"""

import shutil
from pathlib import Path

def clean_directory(path, description):
    """Remove directory if exists."""
    path = Path(path)
    if path.exists():
        shutil.rmtree(path)
        print(f"✅ Видалено: {description} ({path})")
    else:
        print(f"⏭️  Пропущено: {description} (не існує)")

def main():
    """Clean all data."""
    print("=" * 80)
    print("🧹 ОЧИЩЕННЯ ВСІХ ДАНИХ")
    print("=" * 80)
    
    # Clean accumulated data
    print("\n📁 Accumulated data:")
    accumulated_dir = Path("data/colab/accumulated")
    if accumulated_dir.exists():
        for subdir in accumulated_dir.iterdir():
            if subdir.is_dir():
                clean_directory(subdir, f"accumulated/{subdir.name}")
    else:
        print("⏭️  Директорія accumulated не існує")
    
    # Clean cache
    print("\n💾 Cache:")
    clean_directory("data/cache", "cache")
    
    # Clean outputs
    print("\n📤 Outputs:")
    outputs_dir = Path("outputs")
    if outputs_dir.exists():
        for subdir in outputs_dir.iterdir():
            if subdir.is_dir():
                clean_directory(subdir, f"outputs/{subdir.name}")
    else:
        print("⏭️  Директорія outputs не існує")
    
    print("\n" + "=" * 80)
    print("✅ ОЧИЩЕННЯ ЗАВЕРШЕНО")
    print("=" * 80)
    print("\n💡 Тепер можна запустити:")
    print("   python run_hybrid_pipeline.py --mode prepare")

if __name__ == "__main__":
    main()
