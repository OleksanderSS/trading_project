import duckdb
import os

db_path = 'data/raw_data.duckdb'
if os.path.exists(db_path):
    con = duckdb.connect(db_path)
    schema = con.execute("PRAGMA table_info('market_data_raw');").fetchall()
    print("Schema of market_data_raw:")
    for col in schema:
        print(f"- {col[1]} ({col[2]})")
    con.close()
else:
    print(f"DB file not found at {db_path}")
