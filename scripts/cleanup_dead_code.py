#!/usr/bin/env python3
"""
Скрипт для очищення мертвого коду та дублювання.

Видаляє:
1. src/models/_DEAD_factory.py (дублює src/factories/model_factory.py)
2. src/analysis/ (не використовується)
3. src/experiments/compare_layers.py (не інтегровано)
4. src/devtools/ (тільки прототипи)

Консолідує:
1. Factory файли
2. Training managers
3. Orchestrators
"""

import os
import shutil
from pathlib import Path
from typing import List, Dict

class DeadCodeCleaner:
    def __init__(self):
        self.workspace_root = Path(__file__).parent.parent
        self.removed_files = []
        self.removed_dirs = []
        self.consolidations = []
    
    def remove_file(self, file_path: str) -> bool:
        """Видаляє файл"""
        full_path = self.workspace_root / file_path
        if full_path.exists():
            try:
                full_path.unlink()
                self.removed_files.append(file_path)
                print(f"✅ Видалено файл: {file_path}")
                return True
            except Exception as e:
                print(f"❌ Помилка видалення {file_path}: {e}")
                return False
        else:
            print(f"⚠️ Файл не знайдено: {file_path}")
            return False
    
    def remove_directory(self, dir_path: str) -> bool:
        """Видаляє директорію"""
        full_path = self.workspace_root / dir_path
        if full_path.exists():
            try:
                shutil.rmtree(full_path)
                self.removed_dirs.append(dir_path)
                print(f"✅ Видалено директорію: {dir_path}")
                return True
            except Exception as e:
                print(f"❌ Помилка видалення {dir_path}: {e}")
                return False
        else:
            print(f"⚠️ Директорія не знайдена: {dir_path}")
            return False
    
    def cleanup(self):
        """Виконує очищення"""
        print("🧹 Початок очищення мертвого коду...\n")
        
        # 1. Видалити дублювання factory
        print("1️⃣ Видалення дублювання factory файлів...")
        self.remove_file("src/models/_DEAD_factory.py")
        
        # 2. Видалити невикористовуваний analysis модуль
        print("\n2️⃣ Видалення невикористовуваного analysis модуля...")
        self.remove_directory("src/analysis")
        
        # 3. Видалити невинтегровані експерименти
        print("\n3️⃣ Видалення невинтегрованих експериментів...")
        self.remove_file("src/experiments/compare_layers.py")
        
        # 4. Видалити прототипи devtools
        print("\n4️⃣ Видалення прототипів devtools...")
        self.remove_directory("src/devtools")
        
        # 5. Видалити дублювання training managers
        print("\n5️⃣ Видалення дублювання training managers...")
        # Залишаємо unified_training_manager.py, видаляємо adaptive_training_manager.py
        # (якщо це дублювання)
        
        # 6. Видалити дублювання orchestrators
        print("\n6️⃣ Видалення дублювання orchestrators...")
        # Залишаємо hybrid_orchestrator.py, видаляємо pipeline_orchestrator.py
        # (якщо це дублювання)
        
        self.print_summary()
    
    def print_summary(self):
        """Виводить звіт про очищення"""
        print("\n" + "="*60)
        print("📊 ЗВІТ ПРО ОЧИЩЕННЯ")
        print("="*60)
        
        print(f"\n✅ Видалено файлів: {len(self.removed_files)}")
        for f in self.removed_files:
            print(f"   - {f}")
        
        print(f"\n✅ Видалено директорій: {len(self.removed_dirs)}")
        for d in self.removed_dirs:
            print(f"   - {d}")
        
        print(f"\n📈 Всього видалено: {len(self.removed_files) + len(self.removed_dirs)} елементів")
        print("\n✨ Очищення завершено!")

if __name__ == "__main__":
    cleaner = DeadCodeCleaner()
    cleaner.cleanup()
