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

def _validate_directory(base_path):
    """Validate that directory exists."""
    if not base_path.exists():
        print(f"❌ Директорія не існує: {base_path}")
        return False
    return True

def _check_file_eligibility(file_path):
    """Check if file should be processed."""
    return file_path.is_file()

def _is_double_prefix_file(filename, remove_double_prefix):
    """Check if file has double prefix."""
    return remove_double_prefix and has_double_prefix(filename)

def _is_old_file(filename, cutoff_date):
    """Check if file is older than cutoff date."""
    file_timestamp = parse_timestamp_from_filename(filename)
    return file_timestamp and file_timestamp < cutoff_date

def _scan_files(base_path, remove_double_prefix, cutoff_date):
    """Scan directory and categorize files."""
    files_to_delete = []
    double_prefix_files = []
    
    for file_path in base_path.rglob('*'):
        if not _check_file_eligibility(file_path):
            continue
        
        filename = file_path.name
        
        # Check for double prefix
        if _is_double_prefix_file(filename, remove_double_prefix):
            double_prefix_files.append(file_path)
            continue
        
        # Check for old files
        if _is_old_file(filename, cutoff_date):
            files_to_delete.append((file_path, parse_timestamp_from_filename(filename)))
    
    return files_to_delete, double_prefix_files

def _log_statistics(base_path, cutoff_date, dry_run):
    """Log cleanup statistics."""
    print("\n" + "="*80)
    print("📊 СТАТИСТИКА ОЧИЩЕННЯ")
    print("="*80)
    print(f"📁 Директорія: {base_path}")
    print(f"📅 Видалити файли старші за: {cutoff_date.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🔍 Режим: {'DRY RUN (тільки перегляд)' if dry_run else 'ВИДАЛЕННЯ'}")
    print(f"{'='*80}\n")

def _log_double_prefix_files(double_prefix_files):
    """Log files with double prefix."""
    if not double_prefix_files:
        return
    
    print(f"🔴 Файли з подвійним префіксом (target_target_, ticker_ticker_): {len(double_prefix_files)}")
    for i, file_path in enumerate(double_prefix_files[:10], 1):
        print(f"   {i}. {file_path.name}")
    if len(double_prefix_files) > 10:
        print(f"   ... та ще {len(double_prefix_files) - 10} файлів")
    print()

def _log_old_files(files_to_delete, days):
    """Log old files to be deleted."""
    if not files_to_delete:
        return
    
    print(f"⏰ Застарілі файли (старші за {days} днів): {len(files_to_delete)}")
    # Sort by date
    files_to_delete.sort(key=lambda x: x[1])
    for i, (file_path, timestamp) in enumerate(files_to_delete[:10], 1):
        print(f"   {i}. {file_path.name} ({timestamp.strftime('%Y-%m-%d %H:%M')})")
    if len(files_to_delete) > 10:
        print(f"   ... та ще {len(files_to_delete) - 10} файлів")
    print()

def _calculate_file_sizes(double_prefix_files, files_to_delete):
    """Calculate total size of files to be deleted."""
    total_size = 0
    all_files = double_prefix_files + [f[0] for f in files_to_delete]
    for file_path in all_files:
        total_size += file_path.stat().st_size
    
    total_size_mb = total_size / (1024 * 1024)
    return total_size_mb, len(all_files)

def _log_file_summary(total_size_mb, total_files):
    """Log file summary information."""
    print(f"💾 Загальний розмір для видалення: {total_size_mb:.2f} MB")
    print(f"📋 Всього файлів для видалення: {total_files}")
    print()

def _delete_files(all_files):
    """Delete files with progress tracking."""
    print("="*80)
    print("🗑️ ВИДАЛЕННЯ ФАЙЛІВ...")
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
    
    return deleted_count

def _log_dry_run_info(days):
    """Log dry run information."""
    print("="*80)
    print("ℹ️ DRY RUN MODE - файли НЕ видалено")
    print("="*80)
    print("Для видалення запустіть без --dry-run:")
    print("  python scripts/cleanup_old_files.py --days {}".format(days))

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
    
    # Validate directory
    if not _validate_directory(base_path):
        return
    
    cutoff_date = datetime.now() - timedelta(days=days)
    
    # Scan files
    files_to_delete, double_prefix_files = _scan_files(base_path, remove_double_prefix, cutoff_date)
    
    # Log statistics
    _log_statistics(base_path, cutoff_date, dry_run)
    
    # Log specific file types
    _log_double_prefix_files(double_prefix_files)
    _log_old_files(files_to_delete, days)
    
    # Calculate and log summary
    total_size_mb, total_files = _calculate_file_sizes(double_prefix_files, files_to_delete)
    _log_file_summary(total_size_mb, total_files)
    
    # Delete files or show dry run info
    if not dry_run:
        deleted_count = _delete_files(double_prefix_files + [f[0] for f in files_to_delete])
        print(f"\n✅ Видалено {deleted_count} файлів ({total_size_mb:.2f} MB)")
    else:
        _log_dry_run_info(days)
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
