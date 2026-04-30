"""
Перевірка кешу в Colab
"""

import os
from pathlib import Path

def check_colab_cache():
    """Перевірити кеш в Colab"""
    
    # Colab шлях
    colab_cache_path = Path("/content/drive/MyDrive/trading_project/cache")
    colab_unified_cache = colab_cache_path / "unified_cache"
    
    print("🔍 ПЕРЕВІРКА КЕШУ В COLAB")
    print("=" * 50)
    
    # Перевірити основну папку кешу
    if colab_cache_path.exists():
        print(f"✅ Папка кешу існує: {colab_cache_path}")
        
        # Перевірити unified_cache
        if colab_unified_cache.exists():
            files = list(colab_unified_cache.glob("*.pkl"))
            print(f"✅ Unified cache існує: {len(files)} файлів")
            
            if files:
                print("📁 Приклади файлів кешу:")
                for f in files[:5]:  # Показати перші 5
                    print(f"   - {f.name}")
                if len(files) > 5:
                    print(f"   ... та ще {len(files) - 5} файлів")
            else:
                print("📁 Unified cache порожній")
        else:
            print("❌ Unified cache не існує")
    else:
        print("❌ Папка кешу не існує")
    
    # Перевірити SmartFeatureSelector кеш
    try:
        from src.features.selection.smart_selector import SmartFeatureSelector
        if hasattr(SmartFeatureSelector, '_feature_cache'):
            cache_size = len(SmartFeatureSelector._feature_cache)
            print(f"🔧 SmartFeatureSelector._feature_cache: {cache_size} елементів")
        if hasattr(SmartFeatureSelector, '_model_cache'):
            cache_size = len(SmartFeatureSelector._model_cache)
            print(f"🔧 SmartFeatureSelector._model_cache: {cache_size} елементів")
    except Exception as e:
        print(f"❌ Помилка перевірки SmartFeatureSelector: {e}")

if __name__ == "__main__":
    check_colab_cache()
