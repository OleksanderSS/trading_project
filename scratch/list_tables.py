
import duckdb
import os
from src.config.unified_config_manager import UnifiedConfigManager

config = UnifiedConfigManager()
db_path = config.get('paths.raw_db', 'data/database/trading.db')
if os.path.exists(db_path):
    con = duckdb.connect(db_path)
    tables = con.execute('SELECT table_name FROM duckdb_tables()').fetchall()
    print("Tables found:")
    for t in tables:
        print(f" - {t[0]}")
    con.close()
else:
    print(f"Database not found at {db_path}")
