
import duckdb

try:
    con = duckdb.connect(database='data/raw_data.duckdb', read_only=True)
    print("--- Schema for yahoo_finance ---")
    print(con.execute("PRAGMA table_info('yahoo_finance')").fetchall())
    print("\n--- Schema for rss ---")
    print(con.execute("PRAGMA table_info('rss')").fetchall())
    print("\n--- Schema for google_news ---")
    print(con.execute("PRAGMA table_info('google_news')").fetchall())
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    if 'con' in locals() and con:
        con.close()
