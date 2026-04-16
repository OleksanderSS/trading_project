#!/usr/bin/env python3
"""Розділяє великий код на декілька комірок для Jupyter"""

import json
from pathlib import Path

# Читаємо Python файл
with open('colab_clean_cell.py', 'r', encoding='utf-8') as f:
    code = f.read()

# Розділяємо на логічні частини за маркерами
markers = [
    "# ==============================================================================\n# MEMORY MONITORING (NEW)",
    "# ==============================================================================\n# PREMIUM COLAB CELL",
    "# ==============================================================================\n# 1. ПІДКЛЮЧЕННЯ ШЛЯХІВ",
    "# ==============================================================================\n# 2. АВТОМАТИЧНЕ ЗАВАНТАЖЕННЯ",
    "# ==============================================================================\n# 3. СЛУЖБОВІ ФУНКЦІЇ",
    "# ==============================================================================\n# 4. АРХІТЕКТУРИ ВАЖКИХ МОДЕЛЕЙ",
    "# ==============================================================================\n# 5. ЗАВАНТАЖЕННЯ ДАНИХ",
    "# ==============================================================================\n# 5.0 ІНІЦІАЛІЗАЦІЯ КОНФІГ-МЕНЕДЖЕРА",
    "# ==============================================================================\n# 5.1 CONTEXT MAP",
    "# ==============================================================================\n# 6. ДИНАМІЧНЕ ЗАВАНТАЖЕННЯ",
    "# ==============================================================================\n# 7. ГОЛОВНИЙ ЦИКЛ",
    "# ==============================================================================\n# 8. АГРЕГАЦІЯ РЕЗУЛЬТАТІВ",
    "# ==============================================================================\n# 9. SAVE MEMORY LOG",
]

cells = []

for i in range(len(markers)):
    start_marker = markers[i]
    end_marker = markers[i + 1] if i + 1 < len(markers) else None
    
    start_idx = code.find(start_marker)
    if start_idx == -1:
        continue
    
    if end_marker:
        end_idx = code.find(end_marker, start_idx)
        if end_idx == -1:
            cell_code = code[start_idx:]
        else:
            cell_code = code[start_idx:end_idx]
    else:
        cell_code = code[start_idx:]
    
    # Розділяємо на рядки
    lines = cell_code.split('\n')
    
    cells.append({
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': lines
    })

# Створюємо Jupyter notebook з декількома комірками
notebook = {
    'cells': cells,
    'metadata': {
        'kernelspec': {
            'display_name': 'Python 3',
            'language': 'python',
            'name': 'python3'
        },
        'language_info': {
            'codemirror_mode': {
                'name': 'ipython',
                'version': 3
            },
            'file_extension': '.py',
            'mimetype': 'text/x-python',
            'name': 'python',
            'nbconvert_exporter': 'python',
            'pygments_lexer': 'ipython3',
            'version': '3.9.0'
        }
    },
    'nbformat': 4,
    'nbformat_minor': 4
}

# Зберігаємо
with open('colab_clean_cell_split.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f'✅ Notebook з {len(cells)} комірками створено')
print(f'   Файл: colab_clean_cell_split.ipynb')
print()
for i, cell in enumerate(cells, 1):
    lines = len(cell['source'])
    first_line = cell['source'][0][:60] if cell['source'] else "empty"
    print(f'   Комірка {i}: {lines:4d} рядків - {first_line}...')
