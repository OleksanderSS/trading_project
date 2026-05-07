#!/usr/bin/env python3
"""Check 15m data in DuckDB cache."""

import duckdb
from pathlib import Path

cache_db = Path("data/cache/data_cache.db")

if not cache_db.exists():
    print(f"❌ Cache DB not found: {cache_db}")
    exit(1)

print(f"📊 Checking {cache_db}")

conn = duckdb.connect(str(cache_db), read_only=True)

# Check if table exists
tables = conn.execute("SHOW TABLES").fetchall()
print(f"\n📋 Tables: {[t[0] for t in tables]}")

if ('market_data_raw',) in tables:
    # Check 15m data
    result = conn.execute("""
        SELECT interval, COUNT(*) as count, MIN(datetime) as min_date, MAX(datetime) as max_date
        FROM market_data_raw
        WHERE interval = '15m'
        GROUP BY interval
    """).fetchall()
    
    if result:
        print(f"\n✅ 15m data in cache:")
        for row in result:
            print(f"   Interval: {row[0]}")
            print(f"   Count: {row[1]}")
            print(f"   Date range: {row[2]} → {row[3]}")
    else:
        print(f"\n❌ No 15m data in cache")
    
    # Check all intervals
    all_intervals = conn.execute("""
        SELECT interval, COUNT(*) as count
        FROM market_data_raw
        GROUP BY interval
        ORDER BY count DESC
    """).fetchall()
    
    print(f"\n📊 All intervals in cache:")
    for interval, count in all_intervals:
        print(f"   {interval}: {count} rows")
    
    # Check 15m by ticker
    ticker_15m = conn.execute("""
        SELECT ticker, COUNT(*) as count
        FROM market_data_raw
        WHERE interval = '15m'
        GROUP BY ticker
        ORDER BY ticker
    """).fetchall()
    
    if ticker_15m:
        print(f"\n🎯 15m by ticker:")
        for ticker, count in ticker_15m:
            print(f"   {ticker}: {count} rows")
else:
    print(f"\n❌ market_data_raw table not found")

conn.close()
