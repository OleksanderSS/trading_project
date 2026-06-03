import re
from pathlib import Path


def apply_fix_to_file(file_path):
    print(f"Processing: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Regex to find:
    # except Exception as e:
    #     print(...)
    # and replace it with:
    # except Exception as e:
    #     print(...)
    #     raise
    
    # This pattern matches 'except Exception as e:' followed by indented print/log
    # We look for the print statement immediately following
    pattern = r'(except Exception as e:)\n(\s+print\(.*?\))'
    replacement = r'\1\n\2\n        raise'
    
    new_content, count = re.subn(pattern, replacement, content)
    
    if count > 0:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"  Fixed {count} instances.")
    else:
        print("  No instances found.")

target_dir = Path("d:/trading_project/scripts/analysis")
for file_path in target_dir.glob("*.py"):
    # Skip the one we already fixed
    if file_path.name == "analyze_cache.py":
        continue
    apply_fix_to_file(file_path)
