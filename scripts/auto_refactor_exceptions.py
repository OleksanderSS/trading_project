import os
import re

def refactor_broad_exceptions(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Pattern to match: except Exception as e:
    # Capturing the indentation to maintain it
    pattern = r'(\s*)except Exception as e:'
    
    # Replacement: except (ValueError, TypeError, AttributeError, KeyError) as e:
    replacement = r'\1except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:'
    
    new_content = re.sub(pattern, replacement, content)
    
    if new_content != content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"Refactored: {file_path}")
        return True
    return False

# Directories to scan
directories = ['src/algorithms']

for root, _, files in os.walk('src'):
    for file in files:
        if file.endswith('.py'):
            file_path = os.path.join(root, file)
            refactor_broad_exceptions(file_path)
