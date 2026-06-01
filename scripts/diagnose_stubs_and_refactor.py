
import os


def find_stubs():
    stubs = []
    # Directories to scan
    scan_dirs = ['src']
    
    for root, _, files in os.walk('src'):
        for file in files:
            if file.endswith('.py'):
                path = os.path.join(root, file)
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    if 'raise NotImplementedError' in content:
                        stubs.append(path)
    return stubs

def scan_files_for_global_issues():
    issues = {
        'pd_copy': [],
        'string_concat': [],
        'type_hints_missing': []
    }
    
    # Simple heuristic scan
    for root, _, files in os.walk('src'):
        for file in files:
            if file.endswith('.py'):
                path = os.path.join(root, file)
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    if '.copy()' in content:
                        issues['pd_copy'].append(path)
                    # Detect simple string concatenation for paths (heuristic)
                    if '/' in content and 'str' in content:
                        issues['string_concat'].append(path)
    return issues

if __name__ == "__main__":
    print("--- NotImplementedError Stubs ---")
    for stub in find_stubs():
        print(stub)
        
    print("\n--- Potential Refactoring Candidates (Heuristic) ---")
    results = scan_files_for_global_issues()
    print(f"Occurrences of .copy(): {len(results['pd_copy'])}")
    print("Top files for .copy():", results['pd_copy'][:5])
