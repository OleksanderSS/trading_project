
import duckdb

con = duckdb.connect('data/trading_data.duckdb')
tables = con.execute("SELECT table_name FROM duckdb_tables()").fetchall()
print("Table counts:")
for t in tables:
    count = con.execute("SELECT COUNT(*) FROM ?", (t[0],)).fetchone()[0]
    print(f" - {t[0]}: {count}")
con.close()
