from pathlib import Path

path = Path('src/data/management/data_manager.py')
text = path.read_text(encoding='utf-8')
lines = text.splitlines()
search_line = '    _connections: Dict[str, duckdb.DuckDBPyConnection] = {}'
start_index = next((i for i, line in enumerate(lines[1000:], start=1000) if line == search_line), None)
if start_index is None:
    raise SystemExit('Duplicate marker not found after line 1000')
path.write_text('\n'.join(lines[:start_index]) + '\n', encoding='utf-8')
print(f'Removed duplicate data manager block starting at line {start_index + 1}')
