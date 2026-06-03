
import json
from pathlib import Path

notebook_path = Path("d:/trading_project/colab_clean_cell.ipynb")

with open(notebook_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Find code cell and replace the _train_* functions
updated = False

NEW_TRAIN_FUNCTIONS = """    def _train_mlp(self, x_t, x_v, y_t, y_v, ticker, target):
        from sklearn.neural_network import MLPRegressor
        import joblib
        model = MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=self.config_loader.REDUCED_EPOCHS)
        model.fit(x_t, y_t)
        path = self.path_manager.batch_dir / f"model_{ticker}_{target}_mlp.pkl"
        joblib.dump(model, path)
        print(f"      💾 Збережено: {path.name}")
        return {'mse': 0.0}

    def _train_cnn(self, x_t, x_v, y_t, y_v, ticker, target):
        import tensorflow as tf
        x_t_r = x_t.values.reshape(x_t.shape[0], x_t.shape[1], 1)
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(x_t.shape[1], 1)),
            tf.keras.layers.Conv1D(32, 3, activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        model.fit(x_t_r, y_t, epochs=self.config_loader.REDUCED_EPOCHS, verbose=0)
        path = self.path_manager.batch_dir / f"model_{ticker}_{target}_cnn.keras"
        model.save(path)
        print(f"      💾 Збережено: {path.name}")
        return {'loss': 0.0}

    def _train_lstm(self, x_t, x_v, y_t, y_v, ticker, target):
        import tensorflow as tf
        x_t_r = x_t.values.reshape(x_t.shape[0], 1, x_t.shape[1])
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(1, x_t.shape[1])),
            tf.keras.layers.LSTM(64),
            tf.keras.layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        model.fit(x_t_r, y_t, epochs=self.config_loader.REDUCED_EPOCHS, verbose=0)
        path = self.path_manager.batch_dir / f"model_{ticker}_{target}_lstm.keras"
        model.save(path)
        print(f"      💾 Збережено: {path.name}")
        return {'loss': 0.0}

    def _train_gru(self, x_t, x_v, y_t, y_v, ticker, target):
        import tensorflow as tf
        x_t_r = x_t.values.reshape(x_t.shape[0], 1, x_t.shape[1])
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(1, x_t.shape[1])),
            tf.keras.layers.GRU(64),
            tf.keras.layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        model.fit(x_t_r, y_t, epochs=self.config_loader.REDUCED_EPOCHS, verbose=0)
        path = self.path_manager.batch_dir / f"model_{ticker}_{target}_gru.keras"
        model.save(path)
        print(f"      💾 Збережено: {path.name}")
        return {'loss': 0.0}

    def _train_transformer(self, x_t, x_v, y_t, y_v, ticker, target):
        import tensorflow as tf
        x_t_r = x_t.values.reshape(x_t.shape[0], 1, x_t.shape[1])
        inputs = tf.keras.layers.Input(shape=(1, x_t.shape[1]))
        att = tf.keras.layers.MultiHeadAttention(num_heads=2, key_dim=16)(inputs, inputs)
        outputs = tf.keras.layers.Dense(1)(tf.keras.layers.GlobalAveragePooling1D()(att))
        model = tf.keras.Model(inputs, outputs)
        model.compile(optimizer='adam', loss='mse')
        model.fit(x_t_r, y_t, epochs=self.config_loader.REDUCED_EPOCHS, verbose=0)
        path = self.path_manager.batch_dir / f"model_{ticker}_{target}_transformer.keras"
        model.save(path)
        print(f"      💾 Збережено: {path.name}")
        return {'loss': 0.0}

    def _train_tabnet(self, x_t, x_v, y_t, y_v, ticker, target):
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
        path = self.path_manager.batch_dir / f"model_{ticker}_{target}_tabnet"
        model.save_model(str(path))
        print(f"      💾 Збережено: {path.name}.zip")
        return {'loss': 0.0}

    def _train_autoencoder(self, x_t, x_v, y_t, y_v, ticker, target):
        import tensorflow as tf
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(x_t.shape[1],)),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        model.fit(x_t, y_t, epochs=self.config_loader.REDUCED_EPOCHS, verbose=0)
        path = self.path_manager.batch_dir / f"model_{ticker}_{target}_autoencoder.keras"
        model.save(path)
        print(f"      💾 Збережено: {path.name}")
        return {'loss': 0.0}"""

for cell in data.get('cells', []):
    if cell.get('cell_type') != 'code':
        continue
    source = cell.get('source', [])
    
    # Locate where def _train_mlp starts and def _train_autoencoder ends
    start_idx = None
    end_idx = None
    for i, line in enumerate(source):
        if 'def _train_mlp' in line:
            start_idx = i
        if start_idx is not None and 'def _train_autoencoder' in line:
            # Look for the return statement of autoencoder
            for j in range(i, len(source)):
                if "return {'loss': 0.0}" in source[j]:
                    end_idx = j
                    break
            break
            
    if start_idx is not None and end_idx is not None:
        new_lines = [line + '\n' for line in NEW_TRAIN_FUNCTIONS.splitlines()]
        new_lines[-1] = new_lines[-1].rstrip('\n') + '\n'
        cell['source'] = source[:start_idx] + new_lines + source[end_idx+1:]
        updated = True
        print(f"Updated notebook cell with verbose train methods (lines {start_idx}-{end_idx})")

if updated:
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=1)
    print("Notebook successfully updated!")
else:
    print("Could not find the target code block in notebook cells!")
