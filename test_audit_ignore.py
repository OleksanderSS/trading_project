from pathlib import Path
from audit.engine.deep_static_audit import SourceFile

# Test file
path = Path("src/data/collectors/bigquery_collector.py")
root = Path("src")
sf = SourceFile(path, root)

# Check line 44, where the except statement now is
print(f"Line 44: {sf.line(44)}")
print(f"Is ignored P1 BROAD_EXCEPTION_SILENT_RETURN: {sf.is_ignored(44, 'BROAD_EXCEPTION_SILENT_RETURN')}")
