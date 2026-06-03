import os
import re

def find_and_comment_silent_pass(root_dir):
    pattern = re.compile(r"except\s+Exception\s*:\s*pass", re.IGNORECASE)
    
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".py"):
                path = os.path.join(root, file)
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
                
                if pattern.search(content):
                    print(f"Found silent pass in: {path}")
                    # Тут ми можемо запропонувати заміну, 
                    # або просто знайти їх для ручного перегляду.
                    # Закоментування автоматично може бути ризикованим.
                    
if __name__ == "__main__":
    find_and_comment_silent_pass("src")
