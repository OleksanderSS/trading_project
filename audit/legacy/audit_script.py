import glob
import os
import re
import sys

sys.stdout.reconfigure(encoding='utf-8')

files_to_check = []
for f in glob.glob('src/**/*.py', recursive=True):
    if '__pycache__' not in f and 'trained_models' not in f:
        files_to_check.append(f)
print(f'Total files to analyze: {len(files_to_check)}')

# 1. SYNTAX ERRORS
import py_compile

syntax_errors = []
for f in files_to_check:
    try:
        py_compile.compile(f, doraise=True)
    except py_compile.PyCompileError as e:
        syntax_errors.append(str(e))

print('\n=== 1. SYNTAX ERRORS ===')
for e in syntax_errors:
    print(e)

# 2. IMPORT ISSUES
import_issues = []
for f in files_to_check:
    try:
        with open(f, 'r', encoding='utf-8', errors='ignore') as fh:
            for i, line in enumerate(fh, 1):
                stripped = line.strip()
                if stripped.startswith('from src.') or stripped.startswith('import src.'):
                    parts = stripped.replace('from ', '').replace('import ', '').split('.')
                    module_path = 'src/' + '/'.join(parts[:-1]) if len(parts) > 1 else 'src/'
                    module_file = module_path + '.py'
                    module_init = module_path + '/__init__.py'
                    if not os.path.exists(module_file) and not os.path.exists(module_init):
                        import_issues.append(f'{f}:{i} - Module may not exist: {stripped}')
    except:
        pass

print('\n=== 2. IMPORT ISSUES ===')
for issue in import_issues[:30]:
    print(issue)

# 3. DUPLICATE CODE - duplicate imports
duplicate_imports = []
for f in files_to_check:
    try:
        with open(f, 'r', encoding='utf-8', errors='ignore') as fh:
            lines = fh.readlines()
            imports = {}
            for i, line in enumerate(lines, 1):
                stripped = line.strip()
                if stripped.startswith('import ') or stripped.startswith('from '):
                    if stripped in imports:
                        duplicate_imports.append(f'{f}:{i} - Duplicate import: {stripped} (first at line {imports[stripped]})')
                    else:
                        imports[stripped] = i
    except:
        pass

print('\n=== 3. DUPLICATE CODE (Duplicate Imports) ===')
for d in duplicate_imports:
    print(d)

# 4. DEAD CODE - unused imports
dead_code = []
for f in files_to_check:
    try:
        with open(f, 'r', encoding='utf-8', errors='ignore') as fh:
            content = fh.read()
            lines = content.split('\n')
            for i, line in enumerate(lines, 1):
                stripped = line.strip()
                if stripped.startswith('import '):
                    mod = stripped.split('import ')[1].strip().split(' as ')[-1].split(',')[0].strip()
                    if mod and content.count(mod) == 1:
                        dead_code.append(f'{f}:{i} - Possibly unused import: {stripped}')
                elif stripped.startswith('from ') and ' import ' in stripped:
                    parts = stripped.split(' import ')
                    if len(parts) == 2:
                        names = [n.strip().split(' as ')[-1].strip() for n in parts[1].split(',')]
                        for name in names:
                            if name and content.count(name) == 1 and name != '*':
                                dead_code.append(f'{f}:{i} - Possibly unused import: {name}')
    except:
        pass

print('\n=== 4. DEAD CODE (Unused Imports) ===')
for d in dead_code[:50]:
    print(d)

# 5. SECURITY
secret_patterns = [
    (r'password\s*=\s*["\'][^"\']+["\']', 'hardcoded_password'),
    (r'api_key\s*=\s*["\'][^"\']+["\']', 'hardcoded_api_key'),
    (r'secret\s*=\s*["\'][^"\']+["\']', 'hardcoded_secret'),
]
secrets_found = []
for f in files_to_check:
    try:
        with open(f, 'r', encoding='utf-8', errors='ignore') as fh:
            for i, line in enumerate(fh, 1):
                for pat, typ in secret_patterns:
                    if re.search(pat, line, re.IGNORECASE):
                        stripped = line.strip()
                        if 'example' not in stripped.lower() and 'your_' not in stripped.lower() and 'placeholder' not in stripped.lower():
                            secrets_found.append(f'{f}:{i} [{typ}] {stripped[:120]}')
    except:
        pass

print('\n=== 5. SECURITY ISSUES ===')
for s in secrets_found:
    print(s)
if not secrets_found:
    print('No hardcoded secrets found')

# 6. ERROR HANDLING - bare except
bare_excepts = []
for f in files_to_check:
    try:
        with open(f, 'r', encoding='utf-8', errors='ignore') as fh:
            lines = fh.readlines()
            for i, line in enumerate(lines, 1):
                stripped = line.strip()
                if stripped == 'except:' or stripped.startswith('except :'):
                    bare_excepts.append(f'{f}:{i} - Bare except clause')
                elif re.match(r'^except\s*:', stripped):
                    bare_excepts.append(f'{f}:{i} - Bare except clause')
    except:
        pass

print('\n=== 8. ERROR HANDLING (Bare Except) ===')
for b in bare_excepts[:50]:
    print(b)

# 7. PERFORMANCE - append in loops
perf_issues = []
for f in files_to_check:
    try:
        with open(f, 'r', encoding='utf-8', errors='ignore') as fh:
            lines = fh.readlines()
            in_for = False
            for i, line in enumerate(lines, 1):
                stripped = line.strip()
                if stripped.startswith('for ') and ':' in stripped:
                    in_for = True
                    for_indent = len(line) - len(line.lstrip())
                elif in_for:
                    current_indent = len(line) - len(line.lstrip()) if line.strip() else for_indent + 1
                    if current_indent > for_indent and '.append(' in stripped:
                        perf_issues.append(f'{f}:{i} - .append() inside loop')
                    elif stripped and current_indent <= for_indent:
                        in_for = False
    except:
        pass

print('\n=== 9. PERFORMANCE (Append in Loops) ===')
for p in perf_issues[:50]:
    print(p)

# 8. CODE QUALITY - long functions (>50 lines)
long_functions = []
for f in files_to_check:
    try:
        with open(f, 'r', encoding='utf-8', errors='ignore') as fh:
            lines = fh.readlines()
            func_start = None
            func_name = None
            func_indent = 0
            for i, line in enumerate(lines, 1):
                stripped = line.strip()
                if stripped.startswith('def '):
                    if func_start and func_name:
                        length = i - func_start
                        if length > 80:
                            long_functions.append(f'{f}:{func_start} - Function {func_name} is {length} lines long')
                    func_start = i
                    func_name = stripped.split('def ')[1].split('(')[0].strip()
                    func_indent = len(line) - len(line.lstrip())
                elif func_start and stripped and not stripped.startswith('#'):
                    current_indent = len(line) - len(line.lstrip())
                    if current_indent <= func_indent and not stripped.startswith('@'):
                        length = i - func_start
                        if length > 80:
                            long_functions.append(f'{f}:{func_start} - Function {func_name} is {length} lines long')
                        func_start = None
                        func_name = None
    except:
        pass

print('\n=== 10. CODE QUALITY (Long Functions >80 lines) ===')
for lf in long_functions[:50]:
    print(lf)

# Long files
long_files = []
for f in files_to_check:
    try:
        with open(f, 'r', encoding='utf-8', errors='ignore') as fh:
            line_count = sum(1 for _ in fh)
            if line_count > 500:
                long_files.append(f'{f} - {line_count} lines')
    except:
        pass

print('\n=== 10. CODE QUALITY (Long Files >500 lines) ===')
for lf in long_files:
    print(lf)

print('\n=== AUDIT COMPLETE ===')
print(f'Total files checked: {len(files_to_check)}')
print(f'Syntax errors: {len(syntax_errors)}')
print(f'Import issues: {len(import_issues)}')
print(f'Duplicate imports: {len(duplicate_imports)}')
print(f'Possibly unused imports: {len(dead_code)}')
print(f'Security issues: {len(secrets_found)}')
print(f'Bare excepts: {len(bare_excepts)}')
print(f'Performance issues: {len(perf_issues)}')
print(f'Long functions: {len(long_functions)}')
print(f'Long files: {len(long_files)}')
