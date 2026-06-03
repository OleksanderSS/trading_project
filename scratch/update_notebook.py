
import json
import sys
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

notebook_path = Path("d:/trading_project/colab_clean_cell.ipynb")

with open(notebook_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

NEW_TRAIN_TABNET_BLOCK = """    def _train_tabnet(self, x_t, x_v, y_t, y_v, ticker, target):
        try:
            from pytorch_tabnet.tab_model import TabNetRegressor
        except ImportError:
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "pytorch-tabnet"])
            from pytorch_tabnet.tab_model import TabNetRegressor

        x_np = x_t.values.astype(np.float32) if hasattr(x_t, 'values') else np.asarray(x_t, dtype=np.float32)
        y_np = y_t.values.astype(np.float32).reshape(-1, 1) if hasattr(y_t, 'values') else np.asarray(y_t, dtype=np.float32).reshape(-1, 1)
        x_np = np.nan_to_num(x_np, nan=0.0, posinf=0.0, neginf=0.0)
        y_np = np.nan_to_num(y_np, nan=0.0, posinf=0.0, neginf=0.0)

        model = TabNetRegressor(verbose=0)
        model.fit(x_np, y_np, max_epochs=self.config_loader.REDUCED_EPOCHS or 50)
        model.save_model(str(self.path_manager.batch_dir / f"model_{ticker}_{target}_tabnet"))
        return {'loss': 0.0}
"""

updated = False
for cell in data.get('cells', []):
    if cell.get('cell_type') != 'code':
        continue
    source = cell.get('source', [])
    
    # Find the _train_tabnet method - look for it line by line
    start_idx = None
    end_idx = None
    for i, line in enumerate(source):
        line_stripped = line.strip()
        if 'def _train_tabnet' in line_stripped:
            start_idx = i
        if start_idx is not None and i > start_idx:
            # Method ends when we hit the return statement
            if line_stripped.startswith("return {'loss':") or line_stripped == "return {'loss': 0.0}":
                end_idx = i
                break
    
    if start_idx is not None and end_idx is not None:
        # Check it's the OLD version (no ImportError guard)
        block = ''.join(source[start_idx:end_idx+1])
        if 'except ImportError' not in block:
            new_lines = [line + '\n' for line in NEW_TRAIN_TABNET_BLOCK.splitlines()]
            # Fix last line (no trailing newline for last item)
            new_lines[-1] = new_lines[-1].rstrip('\n') + '\n'
            cell['source'] = source[:start_idx] + new_lines + source[end_idx+1:]
            updated = True
            print(f"Patched _train_tabnet (lines {start_idx}-{end_idx})")
        else:
            print("_train_tabnet already patched, skipping")

if updated:
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=1)
    print("Saved notebook successfully!")
else:
    print("Nothing to update.")

# Verify
with open(notebook_path, 'r', encoding='utf-8') as f:
    verify_data = json.load(f)
for cell in verify_data.get('cells', []):
    if cell.get('cell_type') == 'code':
        src = ''.join(cell.get('source', []))
        if '_train_tabnet' in src and 'except ImportError' in src:
            print("VERIFY OK: _train_tabnet has ImportError guard")
        elif '_train_tabnet' in src:
            print("VERIFY FAIL: _train_tabnet still missing ImportError guard!")
