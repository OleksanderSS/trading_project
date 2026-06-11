import os
import sys
import importlib.util
import logging
import traceback

# Force UTF-8 output on Windows CP1252 terminals
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Silence all loggers during audit to prevent encoding crashes
logging.disable(logging.CRITICAL)

def audit_imports(directory):
    errors = []
    sys.path.insert(0, os.path.abspath('.'))

    for root, dirs, files in os.walk(directory):
        # Skip archive and migration dirs
        dirs[:] = [d for d in dirs if d not in ('archive', '__pycache__', '.git')]
        for file in sorted(files):
            if file.endswith('.py') and not file.startswith('__'):
                filepath = os.path.join(root, file)
                relpath = os.path.relpath(filepath, '.')
                module_name = relpath.replace(os.path.sep, '.').rstrip('.py')

                try:
                    spec = importlib.util.spec_from_file_location(module_name, filepath)
                    if spec is not None and spec.loader is not None:
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)
                except Exception:
                    errors.append((filepath, sys.exc_info()))
    return errors

if __name__ == '__main__':
    print("Auditing src directory for import-time errors...")
    errors = audit_imports('src')
    if errors:
        print(f"\nFound {len(errors)} import-time error(s):")
        for filepath, exc_info in errors:
            print("-" * 70)
            print(f"FILE: {filepath}")
            traceback.print_exception(*exc_info, limit=5)
        sys.exit(1)
    else:
        print("All files imported successfully!")
        sys.exit(0)
