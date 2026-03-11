
import sys
import os
import runpy

# Додаємо кореневу директорію проекту до шляху Python
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    # Використовуємо runpy для запуску main.py як головного скрипта
    # Це надійний спосіб, який правильно обробляє __name__ == '__main__'
    runpy.run_path('main.py', run_name='__main__')
except ModuleNotFoundError:
    print("Помилка: Модуль не знайдено. Перевірте шляхи.", file=sys.stderr)
    print(f"sys.path: {sys.path}", file=sys.stderr)
    raise
except Exception as e:
    print(f"Виникла неочікувана помилка: {e}", file=sys.stderr)
    raise
