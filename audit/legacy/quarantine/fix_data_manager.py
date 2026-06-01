with open(r'D:\trading_project\src\data\management\data_manager.py', 'r', encoding='utf-8') as f:
    content = f.read()

insertion_point = content.find('class IDatabaseManager:')
if insertion_point != -1:
    new_class = """
class DataManager(IDatabaseManager):
    \"\"\"Implementation of IDatabaseManager using DuckDB.\"\"\"
    def __init__(self, config_manager, error_handler=None):
        self.config_manager = config_manager
        self.con = duckdb.connect(config_manager.get('paths.raw_db', ':memory:'))
    def execute_query(self, query, params=None):
        self.con.execute(query, params)
    def fetch_all(self, query, params=None):
        return self.con.execute(query, params).fetchdf().to_dict('records')
    def fetch_one(self, query, params=None):
        res = self.con.execute(query, params).fetchdf().to_dict('records')
        return res[0] if res else None
    def table_exists(self, table_name):
        return self.con.execute("SELECT table_name FROM duckdb_tables() WHERE table_name = ?", [table_name]).fetchone() is not None
    def get_table_schema(self, table_name):
        return {row[1]: row[2].upper() for row in self.con.execute(f'PRAGMA table_info("{table_name}")').fetchall()}
    def upsert(self, table_name, df, unique_on=None):
        self.con.register('tmp_df', df)
        self.con.execute(f'INSERT INTO "{table_name}" SELECT * FROM tmp_df')
        self.con.unregister('tmp_df')

"""
    new_content = content[:insertion_point] + new_class + content[insertion_point:]
    with open(r'D:\trading_project\src\data\management\data_manager.py', 'w', encoding='utf-8') as f:
        f.write(new_content)
