import duckdb

con = duckdb.connect('data/trading_data.duckdb')

# Check cadence for all intervals
df3 = con.execute('''
    SELECT
        interval,
        MIN(datetime) as min_dt,
        MAX(datetime) as max_dt,
        COUNT(*) as count,
        (MAX(datetime) - MIN(datetime)) / (COUNT(*) - 1) as avg_cadence
    FROM market_data_raw
    GROUP BY interval
    ORDER BY interval
''').df()
print("All intervals cadence analysis:")
print(df3)
print("\n")

# Export non-corrupted data (exclude 1d)
print("Exporting non-corrupted data...")
df_clean = con.execute("SELECT * FROM market_data_raw WHERE interval != '1d'").df()
print(f"Exported {len(df_clean)} rows of clean data")

# Drop the corrupted table
print("Dropping corrupted table...")
con.execute("DROP TABLE IF EXISTS market_data_raw")

# Recreate table with clean data
print("Recreating table with clean data...")
con.execute("CREATE TABLE market_data_raw AS SELECT * FROM df_clean")

# Verify recreation
df_after = con.execute('''
    SELECT
        interval,
        MIN(datetime) as min_dt,
        MAX(datetime) as max_dt,
        COUNT(*) as count
    FROM market_data_raw
    GROUP BY interval
    ORDER BY interval
''').df()
print("\nIntervals after recreation:")
print(df_after)

con.close()
