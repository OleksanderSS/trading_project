#!/usr/bin/env python3
"""
Скрипт для очищення застарілих файлів у data/colab/accumulated/

Видаляє:
1. Файли старші за N днів
2. Файли з подвійним префіксом (target_target_, ticker_ticker_)
3. Дублікати (залишає тільки найновіші версії)
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
import argparse
import re

def parse_timestamp_from_filename(filename):
    """Витягує timestamp з імені файлу (формат: YYYYMMDD_HHMMSS)"""
    match = re.search(r'(\d{8}_\d{6})', filename)
    if match:
        try:
            return datetime.strptime(match.group(1), '%Y%m%d_%H%M%S')
        except ValueError:
            return None
    return None

def has_double_prefix(filename):
    """Перевіряє чи файл має подвійний префікс"""
    return 'target_target_' in filename or 'ticker_ticker_' in filename

def cleanup_old_files(base_dir, days=7, dry_run=True, remove_double_prefix=True):
    """
    Очищає застарілі файли
    
    Args:
        base_dir: Базова директорія для очищення
        days: Видалити файли старші за N днів
        dry_run: Якщо True, тільки показує що буде видалено
        remove_double_prefix: Якщо True, видаляє файли з подвійним префіксом
    """
    base_path = Path(base_dir)
    if not base_path.exists():
        print(f"❌ Директорія не існує: {base_dir}")
        return
    
    cutoff_date = datetime.now() - timedelta(days=days)
    
    files_to_delete = []
    double_prefix_files = []
    
    # Збираємо файли для видалення
    for file_path in base_path.rglob('*'):
        if not file_path.is_file():
            continue
        
        filename = file_path.name
        
        # Перевірка на подвійний префікс
        if remove_double_prefix and has_double_prefix(filename):
            double_prefix_files.append(file_path)
            continue
        
        # Перевірка на застарілість
        file_timestamp = parse_timestamp_from_filename(filename)
        if file_timestamp and file_timestamp < cutoff_date:
            files_to_delete.append((file_path, file_timestamp))
    
    # Виводимо статистику
    print(f"\n{'='*80}")
    print(f"📊 СТАТИСТИКА ОЧИЩЕННЯ")
    print(f"{'='*80}")
    print(f"📁 Директорія: {base_dir}")
    print(f"📅 Видалити файли старші за: {cutoff_date.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🔍 Режим: {'DRY RUN (тільки перегляд)' if dry_run else 'ВИДАЛЕННЯ'}")
    print(f"{'='*80}\n")
    
    # Файли з подвійним префіксом
    if double_prefix_files:
        print(f"🔴 Файли з подвійним префіксом (target_target_, ticker_ticker_): {len(double_prefix_files)}")
        for i, file_path in enumerate(double_prefix_files[:10], 1):
            print(f"   {i}. {file_path.name}")
        if len(double_prefix_files) > 10:
            print(f"   ... та ще {len(double_prefix_files) - 10} файлів")
        print()
    
    # Застарілі файли
    if files_to_delete:
        print(f"⏰ Застарілі файли (старші за {days} днів): {len(files_to_delete)}")
        # Сортуємо за датою
        files_to_delete.sort(key=lambda x: x[1])
        for i, (file_path, timestamp) in enumerate(files_to_delete[:10], 1):
            print(f"   {i}. {file_path.name} ({timestamp.strftime('%Y-%m-%d %H:%M')})")
        if len(files_to_delete) > 10:
            print(f"   ... та ще {len(files_to_delete) - 10} файлів")
        print()
    
    # Підрахунок розміру
    total_size = 0
    all_files = double_prefix_files + [f[0] for f in files_to_delete]
    for file_path in all_files:
        total_size += file_path.stat().st_size
    
    total_size_mb = total_size / (1024 * 1024)
    print(f"💾 Загальний розмір для видалення: {total_size_mb:.2f} MB")
    print(f"📋 Всього файлів для видалення: {len(all_files)}")
    print()
    
    # Видалення
    if not dry_run:
        print(f"{'='*80}")
        print(f"🗑️ ВИДАЛЕННЯ ФАЙЛІВ...")
        print(f"{'='*80}\n")
        
        deleted_count = 0
        for file_path in all_files:
            try:
                file_path.unlink()
                deleted_count += 1
                if deleted_count % 10 == 0:
                    print(f"   Видалено {deleted_count}/{len(all_files)} файлів...")
            except Exception as e:
                print(f"   ❌ Помилка при видаленні {file_path.name}: {e}")
        
        print(f"\n✅ Видалено {deleted_count} файлів ({total_size_mb:.2f} MB)")
    else:
        print(f"{'='*80}")
        print(f"ℹ️ DRY RUN MODE - файли НЕ видалено")
        print(f"{'='*80}")
        print(f"Для видалення запустіть без --dry-run:")
        print(f"  python scripts/cleanup_old_files.py --days {days}")
        print()

def main():
    parser = argparse.ArgumentParser(description='Очищення застарілих файлів')
    parser.add_argument(
        '--days',
        type=int,
        default=7,
        help='Видалити файли старші за N днів (default: 7)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Тільки показати що буде видалено, не видаляти'
    )
    parser.add_argument(
        '--no-double-prefix',
        action='store_true',
        help='Не видаляти файли з подвійним префіксом'
    )
    parser.add_argument(
        '--dir',
        default='data/colab/accumulated',
        help='Директорія для очищення (default: data/colab/accumulated)'
    )
    
    args = parser.parse_args()
    
    cleanup_old_files(
        base_dir=args.dir,
        days=args.days,
        dry_run=args.dry_run,
        remove_double_prefix=not args.no_double_prefix
    )

if __name__ == '__main__':
    main()
