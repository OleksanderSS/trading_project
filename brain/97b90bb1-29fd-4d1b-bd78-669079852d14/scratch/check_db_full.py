
import duckdb
import os
import pandas as pd

db_path = "d:/trading_project/data/trading_data.duckdb"
if os.path.exists(db_path):
    con = duckdb.connect(db_path)
    try:
        # Check market data
        tables = con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        print(f"Tables: {tables}")
        
        if ('market_data',) in tables:
            counts = con.execute("SELECT ticker, interval, count(*) as count, min(datetime), max(datetime) FROM market_data GROUP BY ticker, interval").df()
            print("\nMarket Data Summary:")
            print(counts.to_string())
        
        # Check news
        if ('news',) in tables:
            news_count = con.execute("SELECT count(*) FROM news").fetchone()[0]
            print(f"\nTotal News: {news_count}")
            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        con.close()
else:
    print("Database not found")
