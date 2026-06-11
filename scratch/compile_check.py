import os
import py_compile
import sys

def audit_syntax(directory):
    errors = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                try:
                    py_compile.compile(filepath, doraise=True)
                except py_compile.PyCompileError as e:
                    errors.append((filepath, str(e)))
    return errors

if __name__ == '__main__':
    print("Auditing src directory for compile/syntax errors...")
    errors = audit_syntax('src')
    if errors:
        print(f"Found {len(errors)} compile errors:")
        for filepath, err in errors:
            print("-" * 60)
            print(f"File: {filepath}")
            print(err)
            print("-" * 60)
        sys.exit(1)
    else:
        print("All files compiled successfully!")
        sys.exit(0)
