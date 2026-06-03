#!/usr/bin/env python
"""Перевіряє всі новини в БД."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.unified_config_manager import UnifiedConfigManager
from src.data.management.data_manager import DataManager
from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger(__name__)

# Initialize
config_manager = UnifiedConfigManager()
db_manager = DataManager(config_manager)

# Check all news tables
news_tables = ['rss_news', 'google_news', 'sec_filings']

print("=" * 80)
print("NEWS DATA IN DATABASE")
print("=" * 80)

total_records = 0

for table in news_tables:
    try:
        df = db_manager.fetch_data_from_table(table)
        if df is not None and not df.empty:
            count = len(df)
            total_records += count
            print(f"\n[OK] {table}: {count} records")
            print(f"   Columns: {list(df.columns)}")
            if 'published_date' in df.columns:
                date_col = 'published_date'
            elif 'filingDate' in df.columns:
                date_col = 'filingDate'
            else:
                date_col = 'date'
            if date_col in df.columns:
                print(f"   Date range: {df[date_col].min()} to {df[date_col].max()}")
        else:
            print(f"\n[WARN] {table}: EMPTY")
    except Exception as e:
        logger.error(f"Error checking {table}: {e}", exc_info=True)

print("\n" + "=" * 80)
print(f"TOTAL NEWS RECORDS: {total_records}")
print("=" * 80)
