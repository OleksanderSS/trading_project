
import duckdb

try:
    con = duckdb.connect(database='data/raw_data.duckdb', read_only=True)
    count = con.execute("SELECT COUNT(*) FROM google_news").fetchone()[0]
    print(f"Success! Found {count} records in 'google_news' table.")
except Exception as e:
    print(f"Error connecting to database or querying table: {e}")

