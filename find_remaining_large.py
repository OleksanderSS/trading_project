import os

def find_large_files(directory, min_lines=400):
    large_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        lines = sum(1 for _ in f)
                        if lines > min_lines:
                            large_files.append((lines, file_path))
                except Exception:
                    pass
    
    return sorted(large_files, key=lambda x: x[0], reverse=True)

if __name__ == "__main__":
    src_dir = 'src'
    large_files = find_large_files(src_dir)
    for lines, path in large_files:
        print(f"{lines} lines : {path}")
