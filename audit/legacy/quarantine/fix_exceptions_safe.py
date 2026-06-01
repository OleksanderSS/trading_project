import os
import re

# Регулярний вираз для пошуку блоків except Exception:
# Шукаємо "except Exception:" і наступні рядки, які можуть бути порожніми або містити pass
pattern = re.compile(r"(except Exception:)([\s\S]*?)(?=\n\S|$)")

def fix_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Спроба знайти "тихі" блоки
    def replacer(match):
        except_line = match.group(1)
        body = match.group(2)
        
        # Перевірка чи блок "тихий" (не містить raise або логування)
        if "raise" in body or "logger" in body or "print" in body:
            return except_line + body
            
        # Заміна на безпечне логування
        indent = re.search(r"^(\s*)", except_line).group(1)
        new_body = f"\n{indent}    logger.error('Error occurred', exc_info=True)\n{indent}    raise"
        return except_line + new_body

    new_content = pattern.sub(replacer, content)
    
    if new_content != content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        return True
    return False

# Список файлів для обробки (можна звузити)
target_dirs = ['src/core', 'src/data', 'src/analytics']

for root, _, files in os.walk('src'):
    for file in files:
        if file.endswith('.py'):
            # Тільки якщо файл у цільових директоріях
            if any(root.startswith(d) for d in target_dirs):
                file_path = os.path.join(root, file)
                if fix_file(file_path):
                    print(f"Fixed: {file_path}")
