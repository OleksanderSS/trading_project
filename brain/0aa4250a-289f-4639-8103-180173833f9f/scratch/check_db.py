import duckdb
import os

db_path = 'data/raw_data.duckdb'
if os.path.exists(db_path):
    con = duckdb.connect(db_path)
    tables = con.execute("SHOW TABLES;").fetchall()
    print("Tables in DB:")
    for table in tables:
        print(f"- {table[0]}")
    con.close()
else:
    print(f"DB file not found at {db_path}")
