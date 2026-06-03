
import os

import duckdb

db_path = "d:/trading_project/data/trading_data.duckdb"
if os.path.exists(db_path):
    con = duckdb.connect(db_path)
    try:
        count = con.execute("SELECT count(*) FROM news_sentiment_cache").fetchone()[0]
        print(f"Cache count: {count}")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        con.close()
else:
    print("Database not found")
