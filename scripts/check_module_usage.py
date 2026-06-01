
import ast
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger("check_module_usage")


def get_imports(filepath):
    """Витягує всі імпорти з файлу."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            tree = ast.parse(f.read(), filename=filepath)
        
        imports = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for n in node.names:
                    imports.add(n.name.split('.')[0])
            elif isinstance(node, ast.ImportFrom):
                if node.level == 0: # Absolute import
                    imports.add(node.module.split('.')[0])
        return imports
    except Exception as e:
        logger.error(f"Failed to get imports from {filepath}", exc_info=True)
        return set()

def analyze_project(src_dir='src'):
    src_path = Path(src_dir)
    all_files = list(src_path.rglob('*.py'))
    
    # Створюємо мапу модулів
    module_usage = {str(file.relative_to(src_path)): 0 for file in all_files}
    
    for file in all_files:
        imports = get_imports(file)
        for imp in imports:
            # Перевіряємо, чи цей імпорт веде на файл у нашому проекті
            if (src_path / imp).exists() or (src_path / (imp + ".py")).exists():
                if imp in module_usage:
                    module_usage[imp] += 1
                else:
                    # Можливо це підпапка
                    for mod in module_usage:
                        if mod.startswith(imp):
                            module_usage[mod] += 1
                            
    print(f"{'Модуль':<50} | {'Використано (кількість імпортів)':<30}")
    print("-" * 80)
    for mod, count in sorted(module_usage.items(), key=lambda x: x[1]):
        print(f"{mod:<50} | {count:<30}")

if __name__ == "__main__":
    analyze_project()
