
import duckdb
import os
import pandas as pd

db_path = "d:/trading_project/data/trading_data.duckdb"
if os.path.exists(db_path):
    con = duckdb.connect(db_path)
    try:
        # Check raw market data
        print("Market Data Raw Summary:")
        df_market = con.execute("SELECT ticker, interval, count(*) as count, min(datetime) as start, max(datetime) as end FROM market_data_raw GROUP BY ticker, interval").df()
        print(df_market.to_string())
        
        # Check news
        print("\nNews Summary:")
        news_google = con.execute("SELECT count(*) FROM google_news").fetchone()[0]
        news_rss = con.execute("SELECT count(*) FROM rss_news").fetchone()[0]
        print(f"Google News: {news_google}")
        print(f"RSS News: {news_rss}")
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        con.close()
else:
    print("Database not found")
