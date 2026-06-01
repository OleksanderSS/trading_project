
import os
import re
from collections import Counter

def count_silent_exceptions(root_dir):
    silent_except_pattern = re.compile(
        r"except\s+(?:Exception|BaseException)(?:\s+as\s+\w+)?:\s*$"
        r"|except\s+(?:Exception|BaseException)(?:\s+as\s+\w+)?:\s*\n\s*(?:pass|return|continue|break)", 
        re.MULTILINE
    )
    
    file_counts = Counter()
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".py"):
                path = os.path.join(root, file)
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        content = f.read()
                        matches = list(silent_except_pattern.finditer(content))
                        if matches:
                            file_counts[path] = len(matches)
                except Exception:
                    pass
    return file_counts

if __name__ == "__main__":
    file_counts = count_silent_exceptions("src")
    print("Top files with silent exceptions:")
    for path, count in file_counts.most_common(20):
        print(f"{count} -> {path}")
