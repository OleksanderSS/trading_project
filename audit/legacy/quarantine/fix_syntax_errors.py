import os
import re


def fix_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Патерн для пошуку пошкоджених блоків except
        # Знаходить 'except Exception as e:' або 'except Exception:', за яким слідує 
        # неправильно відступлений logger.error та raise
        pattern = r"(except\s+Exception.*:\n)\s+logger\.error\(f['\"].*['\"]\),\s*exc_info=True\)\n\s+raise"
        
        # Замінюємо на коректний блок
        # Залишаємо лише логування з правильним відступом, якщо потрібно, 
        # або видаляємо пошкоджений код
        if re.search(pattern, content):
            new_content = re.sub(pattern, r"\1        logger.error('Error occurred', exc_info=True)\n        raise", content)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            return True
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
    return False

root_dir = r'D:\trading_project\src'
for root, dirs, files in os.walk(root_dir):
    for file in files:
        if file.endswith('.py'):
            if fix_file(os.path.join(root, file)):
                print(f"Fixed {os.path.join(root, file)}")
