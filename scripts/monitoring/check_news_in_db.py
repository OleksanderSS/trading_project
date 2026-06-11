#!/usr/bin/env python
"""Перевіряє, чи дані новин зберігаються в БД."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import duckdb
import pandas as pd
from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger(__name__)

# Підключаємось до БД
conn = duckdb.connect('data/main_database.duckdb')

# Перевіряємо таблиці
tables = conn.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'").fetchall()
print('📊 ТАБЛИЦІ В БД:')
if tables:
    for table in tables:
        print(f'  - {table[0]}')
else:
    print('  (немає таблиць)')

# Альтернативний запит
all_tables = conn.execute("SELECT * FROM duckdb_tables()").fetchall()
print('\n📊 ВСІ ТАБЛИЦІ (duckdb_tables):')
for table in all_tables:
    print(f'  - {table[0]}')

# Перевіряємо RSS дані
try:
    rss_count = conn.execute('SELECT COUNT(*) FROM rss_news').fetchone()[0]
    print(f'\n✅ rss_news: {rss_count} записів')
    
    # Показуємо перші 3 записи
    rss_sample = conn.execute('SELECT title, source, published_date FROM rss_news LIMIT 3').fetchall()
    print('  Приклади:')
    for row in rss_sample:
        print(f'    - {row[0][:50]}... ({row[1]}, {row[2]})')
except Exception as e:
    logger.error(f"Error checking rss_news: {e}", exc_info=True)
    print(f'❌ rss_news: {e}')

# Перевіряємо Google News
try:
    gn_count = conn.execute('SELECT COUNT(*) FROM google_news').fetchone()[0]
    print(f'\n✅ google_news: {gn_count} записів')
except Exception as e:
    logger.error(f"Error checking google_news: {e}", exc_info=True)
    print(f'❌ google_news: {e}')

# Перевіряємо SEC Filings
try:
    sec_count = conn.execute('SELECT COUNT(*) FROM sec_filings').fetchone()[0]
    print(f'✅ sec_filings: {sec_count} записів')
except Exception as e:
    logger.error(f"Error checking sec_filings: {e}", exc_info=True)
    print(f'❌ sec_filings: {e}')

conn.close()
