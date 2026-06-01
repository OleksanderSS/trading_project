import json

nb = json.load(open('colab_clean_cell.ipynb', encoding='utf-8'))
for cell in nb['cells']:
    if cell.get('cell_type') == 'code':
        src = ''.join(cell.get('source', []))
        if '_train_tabnet' in src:
            lines = src.splitlines()
            in_fn = False
            for line in lines:
                if 'def _train_tabnet' in line:
                    in_fn = True
                if in_fn:
                    print(line)
                    if 'return' in line and 'loss' in line:
                        break
