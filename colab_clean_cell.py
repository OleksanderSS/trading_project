# ==============================================================================
# COLAB TRAINING MODULES - REFACTORED ARCHITECTURE
# ==============================================================================

"""
Рефакторинг colab_clean_cell.py на модульну архітектуру.
Кожен клас відповідає за свою область відповідальності.
"""

import os
import sys
import json
import hashlib
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ⚠️ ВАЖНЫЕ ИМПОРТЫ: Перенесены в функции для избежания зависания при импорте модуля
# - torch, torch.nn
# - pandas, numpy, sklearn
# - psutil и другие тяжелые зависимости

# ==============================================================================
# 1. MEMORY MANAGEMENT MODULE
# ==============================================================================

class MemoryMonitor:
    """Моніторинг та управління пам'яттю в Colab"""

    def __init__(self, warning_threshold=75, critical_threshold=90):
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.memory_log = []
        self.start_time = datetime.now()

    def get_memory_info(self):
        """Отримати детальну інформацію про пам'ять"""
        import psutil
        mem = psutil.virtual_memory()
        return {
            'percent': mem.percent,
            'used_gb': mem.used / (1024**3),
            'available_gb': mem.available / (1024**3),
            'total_gb': mem.total / (1024**3)
        }

    def get_memory_usage(self):
        """Отримати відсоток використання пам'яті"""
        import psutil
        return psutil.virtual_memory().percent

    def check_memory(self, context=""):
        """Перевірити пам'ять та залогувати якщо потрібно"""
        info = self.get_memory_info()
        timestamp = datetime.now().isoformat()

        log_entry = {
            'timestamp': timestamp,
            'context': context,
            'memory_info': info
        }
        self.memory_log.append(log_entry)

        status = 'ok'
        if info['percent'] >= self.critical_threshold:
            print(f"🚨 CRITICAL MEMORY: {info['percent']:.1f}% ({info['used_gb']:.1f}GB / {info['total_gb']:.1f}GB)")
            print(f"   Context: {context}")
            status = 'critical'
        elif info['percent'] >= self.warning_threshold:
            print(f"⚠️ WARNING MEMORY: {info['percent']:.1f}% ({info['used_gb']:.1f}GB / {info['total_gb']:.1f}GB)")
            print(f"   Context: {context}")
            status = 'warning'
        else:
            print(f"✅ Memory OK: {info['percent']:.1f}% ({info['used_gb']:.1f}GB / {info['total_gb']:.1f}GB)")

        return status

    def cleanup(self):
        """Примусова збірка сміття"""
        import gc
        gc.collect()
        print("🧹 Garbage collection triggered")

    def save_log(self, filepath):
        """Зберегти лог пам'яті у файл"""
        with open(filepath, 'w') as f:
            json.dump(self.memory_log, f, indent=2)
        print(f"💾 Memory log saved to {filepath}")

# ==============================================================================
# 2. UTILITY FUNCTIONS MODULE
# ==============================================================================

def get_optimal_batch_size(data_size, memory_percent, base_batch_size=32):
    """
    Розрахувати оптимальний розмір батча на основі доступної пам'яті

    Логіка:
    - Якщо пам'ять < 50%: використовуємо base_batch_size
    - Якщо пам'ять 50-75%: зменшуємо до base_batch_size // 2
    - Якщо пам'ять 75-90%: зменшуємо до base_batch_size // 4
    - Якщо пам'ять > 90%: зменшуємо до base_batch_size // 8 (мінімум 2)
    """
    if memory_percent < 50:
        return base_batch_size
    elif memory_percent < 75:
        return max(base_batch_size // 2, 8)
    elif memory_percent < 90:
        return max(base_batch_size // 4, 4)
    else:
        return max(base_batch_size // 8, 2)

def save_checkpoint(ticker, target_col, m_type, model, optimizer, epoch, loss, checkpoint_dir):
    """Зберегти checkpoint для відновлення тренування"""
    import torch
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': float(loss),
        'timestamp': datetime.now().isoformat()
    }

    checkpoint_path = checkpoint_dir / f"checkpoint_{ticker}_{target_col}_{m_type}_ep{epoch}.pt"
    torch.save(checkpoint, checkpoint_path)
    print(f"   ✅ Checkpoint saved: {checkpoint_path.name}")
    return checkpoint_path

def load_checkpoint(checkpoint_path, model, optimizer):
    """Завантажити checkpoint для відновлення тренування"""
    import torch
    
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"   ✅ Checkpoint loaded from epoch {epoch} (loss: {loss:.6f})")
    return epoch, loss

def find_latest_checkpoint(checkpoint_dir, ticker, target_col, m_type):
    """Знайти найновіший checkpoint для моделі"""
    pattern = f"checkpoint_{ticker}_{target_col}_{m_type}_ep*.pt"
    checkpoints = list(checkpoint_dir.glob(pattern))
    if checkpoints:
        # Сортуємо за номером епохи (спадаючо)
        checkpoints.sort(key=lambda x: int(x.stem.split('_ep')[-1]), reverse=True)
        return checkpoints[0]
    return None

def retry_on_timeout(max_retries=3, wait_seconds=5):
    """Декоратор для повтору при timeout"""
    from functools import wraps
    import time

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except (TimeoutError, ConnectionError, RuntimeError) as e:
                    if attempt < max_retries - 1:
                        print(f"⚠️ Attempt {attempt + 1} failed: {str(e)[:100]}")
                        print(f"   Retrying in {wait_seconds} seconds...")
                        time.sleep(wait_seconds)
                    else:
                        print(f"❌ Failed after {max_retries} attempts")
                        raise
        return wrapper
    return decorator

def compute_data_signature(df_feat, df_targ):
    import pandas as pd
    feat_info = f"{df_feat.shape}_{pd.util.hash_pandas_object(df_feat.tail(100)).sum()}"
    targ_info = f"{df_targ.shape}_{pd.util.hash_pandas_object(df_targ.tail(100)).sum()}"
    combined = f"{feat_info}_{targ_info}"
    return hashlib.md5(combined.encode()).hexdigest()

def compute_metrics(y_true, y_pred):
    import numpy as np
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.sum() > 0 else 0.0
    return {'mae': float(mae), 'rmse': float(rmse), 'r2': float(r2), 'mape': float(mape)}

# ==============================================================================
# 3. MODEL ARCHITECTURES MODULE
# ==============================================================================

def create_model(model_type, input_size):
    import torch
    import torch.nn as nn
    
    if model_type == 'mlp':
        return nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    elif model_type == 'lstm':
        class LSTMModel(nn.Module):
            def __init__(self, input_sz):
                super().__init__()
                self.lstm = nn.LSTM(input_sz, 64, 2, batch_first=True, dropout=0.2)
                self.fc = nn.Linear(64, 1)
            def forward(self, x):
                out, _ = self.lstm(x.unsqueeze(1))
                return self.fc(out[:, -1, :])
        return LSTMModel(input_size)
    elif model_type == 'gru':
        class GRUModel(nn.Module):
            def __init__(self, input_sz):
                super().__init__()
                self.gru = nn.GRU(input_sz, 64, 2, batch_first=True, dropout=0.2)
                self.fc = nn.Linear(64, 1)
            def forward(self, x):
                out, _ = self.gru(x.unsqueeze(1))
                return self.fc(out[:, -1, :])
        return GRUModel(input_size)
    elif model_type == 'cnn':
        class CNNModel(nn.Module):
            def __init__(self, input_sz):
                super().__init__()
                self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
                self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
                self.pool = nn.AdaptiveAvgPool1d(1)
                self.fc = nn.Linear(64, 1)
            def forward(self, x):
                x = x.unsqueeze(1)
                x = torch.relu(self.conv1(x))
                x = torch.relu(self.conv2(x))
                return self.fc(self.pool(x).squeeze(-1))
        return CNNModel(input_size)
    elif model_type == 'transformer':
        class TransformerModel(nn.Module):
            def __init__(self, input_sz):
                super().__init__()
                self.embedding = nn.Linear(input_sz, 64)
                encoder_layer = nn.TransformerEncoderLayer(64, 4, dim_feedforward=128, dropout=0.2, batch_first=True)
                self.transformer = nn.TransformerEncoder(encoder_layer, 2)
                self.fc = nn.Linear(64, 1)
            def forward(self, x):
                x = self.embedding(x.unsqueeze(1))
                x = self.transformer(x)
                return self.fc(x[:, -1, :])
        return TransformerModel(input_size)
    elif model_type == 'tabnet':
        # Fallback для TabNet у Colab щоб уникнути конфліктів залежностей (використовуємо потужний MLP)
        return nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
    elif model_type == 'autoencoder':
        class AutoencoderModel(nn.Module):
            def __init__(self, input_sz):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(input_sz, 64), nn.ReLU(),
                    nn.Linear(64, 32), nn.ReLU()
                )
                self.decoder = nn.Sequential(
                    nn.Linear(32, 16), nn.ReLU(),
                    nn.Linear(16, 1)
                )
            def forward(self, x):
                return self.decoder(self.encoder(x))
        return AutoencoderModel(input_size)
    else:
        # Fallback для будь-якої іншої моделі
        return nn.Sequential(
            nn.Linear(input_size, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1)
        )

# ==============================================================================
# 4. ENVIRONMENT SETUP MODULE
# ==============================================================================

class ColabEnvironment:
    """Налаштування середовища Colab"""

    def __init__(self):
        self.PROJECT_PATH = None
        self.SRC_PATH = None
        self.BATCH_NAME = "main_database"
        self.batch_dir = None
        self.models_dir = None

    def setup_paths(self):
        """Налаштування шляхів для Colab або локального середовища"""
        try:
            from google.colab import drive
            import signal
            
            # Таймаут 5 секунд для попытки подключения
            def timeout_handler(signum, frame):
                raise TimeoutError("Таймаут при подключении к Google Colab")
            
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(5)
            
            try:
                drive.mount('/content/drive')
                signal.alarm(0)  # Отмена таймаута
                self.PROJECT_PATH = '/content/drive/MyDrive/trading_project'
                print(f"✅ Google Drive підключено")
            except TimeoutError:
                signal.alarm(0)  # Отмена таймаута
                print("⚠️ Таймаут при подключении к Google Drive, працюємо локально")
                raise ImportError("Таймаут Colab")
        except (ImportError, TimeoutError, Exception) as e:
            self.PROJECT_PATH = str(Path.cwd())
            print("⚠️ Працюємо локально")

        self.SRC_PATH = os.path.join(self.PROJECT_PATH, "src")
        for p in [self.PROJECT_PATH, self.SRC_PATH]:
            if p not in sys.path:
                sys.path.insert(0, p)

        print(f"📁 PROJECT_PATH: {self.PROJECT_PATH}")
        print(f"📁 Current dir: {os.getcwd()}")

    def setup_batch_directory(self, batch_name=None):
        """Налаштування директорії для батча"""
        if batch_name:
            self.BATCH_NAME = batch_name

        self.batch_dir = Path(self.PROJECT_PATH) / "data" / "colab" / "accumulated" / self.BATCH_NAME
        self.models_dir = self.batch_dir / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)

        return self.batch_dir, self.models_dir

# ==============================================================================
# 5. CONFIGURATION LOADER MODULE
# ==============================================================================

class RuntimeConfigLoader:
    """Завантаження та управління runtime конфігурацією"""

    def __init__(self, project_path):
        self.project_path = Path(project_path)
        self.runtime_params = {}
        self.test_mode = {}

        # Тестові параметри за замовчуванням
        self.TEST_TICKER = None
        self.TEST_TARGET = None
        self.REDUCED_EPOCHS = None
        self.MAX_ITERATIONS = 100

    def load_runtime_params(self, batch_dir):
        """Завантаження runtime параметрів з файлів"""
        # Спочатку шукаємо centralized runtime params, потім legacy src/config
        runtime_params_path = self.project_path / "data" / "runtime" / "runtime_params.json"
        legacy_runtime_params_path = self.project_path / "src" / "config" / "runtime_params.json"

        if not runtime_params_path.exists() and legacy_runtime_params_path.exists():
            runtime_params_path = legacy_runtime_params_path

        if runtime_params_path.exists():
            with open(runtime_params_path, 'r') as f:
                temp_params = json.load(f)
                batch_name = temp_params.get('batch', {}).get('batch_name', 'main_database')

                # Уніфікація назв
                batch_name = batch_name.replace('target_target_', 'target_')
                if not batch_name.startswith('test_') and batch_name != 'main_database':
                    batch_name = 'test_' + batch_name

                if runtime_params_path == legacy_runtime_params_path:
                    print(f"⚠️ Batch name з legacy src/config/runtime_params.json: {batch_name}")
                else:
                    print(f"📦 Batch name з data/runtime/runtime_params.json: {batch_name}")
                batch_dir = self.project_path / "data" / "colab" / "accumulated" / batch_name

        # Шукаємо runtime_params.json в batch директорії
        runtime_params_path = batch_dir / "runtime_params.json"
        if runtime_params_path.exists():
            with open(runtime_params_path, 'r') as f:
                self.runtime_params = json.load(f)

            self.test_mode = self.runtime_params.get('test_mode', {})
            self.TEST_TICKER = self.test_mode.get('test_ticker')
            self.TEST_TARGET = self.test_mode.get('test_target')
            self.REDUCED_EPOCHS = self.test_mode.get('reduced_epochs')
            self.MAX_ITERATIONS = self.runtime_params.get('models', {}).get('max_iterations', 100)

            self._print_loaded_params()
        else:
            print("⚠️ runtime_params.json не знайдено, використовуємо параметри за замовчуванням")

    def _print_loaded_params(self):
        """Вивід завантажених параметрів"""
        print("\n" + "="*80)
        print("📥 ПАРАМЕТРИ ЗАВАНТАЖЕНО З runtime_params.json")
        print("="*80)
        print(f"  Режим: {self.runtime_params.get('mode', 'full')}")

        tickers_list = self.runtime_params.get('training', {}).get('tickers', [])
        print(f"  Тікери: {tickers_list if tickers_list else 'всі з конфігу'}")

        timeframes_list = self.runtime_params.get('training', {}).get('timeframes', [])
        if not timeframes_list:
            timeframes_list = ['15m', '1h', '1d']
            print(f"  ⚠️ Таймфрейми пусті в runtime_params, використовуємо дефолтні: {timeframes_list}")
        else:
            print(f"  Таймфрейми: {timeframes_list}")

        if self.TEST_TICKER:
            print(f"  🧪 Тестовий тікер: {self.TEST_TICKER}")
        if self.TEST_TARGET:
            print(f"  🧪 Тестовий таргет: {self.TEST_TARGET}")
        if self.REDUCED_EPOCHS:
            print(f"  ⚡ Епохи: {self.REDUCED_EPOCHS} (замість 50)")
        if self.MAX_ITERATIONS != 100:
            print(f"  ⚡ Ітерації: {self.MAX_ITERATIONS} (замість 100)")
        print("="*80 + "\n")

# ==============================================================================
# 6. DATA LOADER MODULE
# ==============================================================================

class ColabDataLoader:
    """Завантаження та підготовка даних для Colab"""

    def __init__(self, batch_dir, config_loader):
        self.batch_dir = batch_dir
        self.config_loader = config_loader
        self.features_df = None
        self.targets_df = None
        self.data_signature = None

    def load_data(self):
        """Завантаження features та targets"""
        import pandas as pd
        
        features_path = self.batch_dir / "features.parquet"
        targets_path = self.batch_dir / "targets.parquet"

        if not features_path.exists() or not targets_path.exists():
            raise FileNotFoundError(f"Файли відсутні в {self.batch_dir}")

        self.features_df = pd.read_parquet(features_path)
        self.targets_df = pd.read_parquet(targets_path)

        # Видаляємо target колонки з features якщо вони є
        target_cols_in_features = [c for c in self.features_df.columns
                                 if c in self.targets_df.columns and c not in ['ticker', 'datetime', 'interval']]
        if target_cols_in_features:
            print(f"⚠️ Знайдено {len(target_cols_in_features)} target колонок у features.parquet: {target_cols_in_features}")
            print("   Видаляємо їх щоб уникнути конфлікту імен при merge...")
            self.features_df = self.features_df.drop(columns=target_cols_in_features)
            print(f"   ✅ Видалено. Нова форма features: {self.features_df.shape}")

        print(f"✅ Базу завантажено: Features {self.features_df.shape}, Targets {self.targets_df.shape}")

        # Нормалізація timezone
        self._normalize_timezones()

        return self.features_df, self.targets_df

    def _normalize_timezones(self):
        """Нормалізація timezone для merge"""
        import pandas as pd
        
        if 'datetime' in self.features_df.columns:
            self.features_df['datetime'] = pd.to_datetime(self.features_df['datetime']).dt.tz_localize(None)
        if 'datetime' in self.targets_df.columns:
            self.targets_df['datetime'] = pd.to_datetime(self.targets_df['datetime']).dt.tz_localize(None)
        print("✅ Timezone нормалізовано для обох датафреймів")

    def check_cache(self):
        """Перевірка кешу даних"""
        self.data_signature = compute_data_signature(self.features_df, self.targets_df)
        cache_file = self.batch_dir / "colab_cache_sig.json"

        if cache_file.exists():
            with open(cache_file, 'r') as f:
                cached_data = json.load(f)
                if cached_data.get('signature') == self.data_signature:
                    print("\n✅ ДАНІ НЕ ЗМІНИЛИСЯ. Пропускаємо глобальний перерахунок (доучуємо залишки).")
                    return True
                else:
                    print("\n⚠️ Дані змінилися! Глобальний кеш скинуто.")
                    return False
        else:
            print("\n🆕 Нова сесія. Кеш бази відсутній.")
            return False

    def save_cache_signature(self):
        """Збереження сигнатури кешу"""
        cache_file = self.batch_dir / "colab_cache_sig.json"
        with open(cache_file, 'w') as f:
            json.dump({'signature': self.data_signature, 'date': datetime.now().isoformat()}, f)

# ==============================================================================
# 7. FEATURE SELECTOR MODULE
# ==============================================================================

class ColabFeatureSelector:
    """Вибір фіч для моделей у Colab"""

    def __init__(self, project_path):
        self.project_path = Path(project_path)
        self.feature_selector = None
        self._init_selector()

    def _init_selector(self):
        """Ініціалізація SmartFeatureSelector"""
        try:
            from src.features.selection.smart_selector import SmartFeatureSelector

            # Use centralized cache path: data/cache/selected_features.json (not src/config)
            cache_path = self.project_path / "data" / "cache" / "selected_features.json"
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.feature_selector = SmartFeatureSelector(
                storage_path=str(cache_path)
            )
            print("✅ SmartFeatureSelector ініціалізовано")
        except Exception as e:
            print(f"❌ Помилка при ініціалізації SmartFeatureSelector: {e}")
            raise

    def select_features(self, X, y, context_id, is_classification=False, max_features=None):
        """Вибір фіч для моделі"""
        return self.feature_selector.select(
            X=X,
            y=y,
            context_id=context_id,
            is_classification=is_classification,
            market_regime="normal",
            force_recalculate=False,
            max_features=max_features
        )

# ==============================================================================
# 8. MODEL TRAINER MODULE
# ==============================================================================

class HeavyModelTrainer:
    """Тренування важких моделей у Colab"""

    def __init__(self, batch_dir, memory_monitor, config_loader):
        self.batch_dir = batch_dir
        self.memory_monitor = memory_monitor
        self.config_loader = config_loader
        self.checkpoint_dir = batch_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)

    def train_model(self, ticker, target_col, model_type, X_df, y_ser, available_features):
        """Тренування однієї моделі"""
        # ⚠️ Ленивые импорты
        import pandas as pd
        import numpy as np
        import torch
        import torch.nn as nn
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        
        print(f"    🔍 {model_type:<14} | ", end="")

        try:
            # Підготовка даних
            X_vals = X_df[available_features].values
            y_vals = y_ser.values

            # Deduplication для швидкості
            combined_data = pd.DataFrame(X_df[available_features])
            combined_data['__target__'] = y_ser.values
            original_len = len(combined_data)
            unique_data = combined_data.drop_duplicates()
            unique_len = len(unique_data)

            if unique_len < original_len:
                print(f"⚡ Dedup: {original_len} -> {unique_len} ({100*(1-unique_len/original_len):.1f}% reduction) | ", end="")
                X_vals = unique_data.drop(columns=['__target__']).values
                y_vals = unique_data['__target__'].values

            # Train/test split - TIME-BASED (без shuffle для часових рядів!)
            split_idx = int(len(X_vals) * 0.8)
            X_tr, X_va = X_vals[:split_idx], X_vals[split_idx:]
            y_tr, y_va = y_vals[:split_idx], y_vals[split_idx:]

            # ✅ ВИПРАВЛЕНО: Масштабуємо ЯК X, ТАК І Y
            # X scaler
            scaler = StandardScaler()
            X_tr_sc = scaler.fit_transform(X_tr)
            X_va_sc = scaler.transform(X_va)

            # Y scaler (НОВОЕ!)
            y_scaler = StandardScaler()
            y_tr_sc = y_scaler.fit_transform(y_tr.reshape(-1, 1)).flatten()
            y_va_sc = y_scaler.transform(y_va.reshape(-1, 1)).flatten()

            # PyTorch tensors - використовуємо масштабовані Y
            X_tr_t = torch.FloatTensor(X_tr_sc)
            y_tr_t = torch.FloatTensor(y_tr_sc).reshape(-1, 1)  # ✅ Масштабований Y
            X_va_t = torch.FloatTensor(X_va_sc)
            y_va_t = torch.FloatTensor(y_va_sc).reshape(-1, 1)  # ✅ Масштабований Y

            # Створення моделі
            model = create_model(model_type, len(available_features))
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

            # Параметри тренування
            epochs = self.config_loader.REDUCED_EPOCHS if self.config_loader.REDUCED_EPOCHS else 50
            base_batch_size = 32
            patience, patience_counter = 10, 0
            best_loss = float('inf')
            best_model_state = None

            # Checkpoint recovery
            start_epoch = 0
            latest_checkpoint = find_latest_checkpoint(self.checkpoint_dir, ticker, target_col, model_type)
            if latest_checkpoint:
                print(f"🔄 Found checkpoint: {latest_checkpoint.name}")
                start_epoch, best_loss = load_checkpoint(latest_checkpoint, model, optimizer)
                start_epoch += 1

            # Training loop
            for ep in range(start_epoch, epochs):
                # Memory check
                if ep % 2 == 0:
                    memory_status = self.memory_monitor.check_memory(f"Epoch {ep} - {model_type}")
                    if memory_status == 'critical':
                        print("   ⚠️ Critical memory, stopping training")
                        break

                # Optimal batch size
                current_memory = self.memory_monitor.get_memory_usage()
                batch_size = get_optimal_batch_size(len(X_vals), current_memory, base_batch_size=base_batch_size)

                model.train()
                for i in range(0, len(X_tr_t), batch_size):
                    batch_X = X_tr_t[i:i+batch_size]
                    batch_y = y_tr_t[i:i+batch_size]

                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()

                # Validation
                model.eval()
                with torch.no_grad():
                    val_outputs = model(X_va_t)
                    val_loss = criterion(val_outputs, y_va_t).item()

                if val_loss < best_loss:
                    best_loss = val_loss
                    patience_counter = 0
                    best_model_state = model.state_dict()
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        break

                # Checkpoint save
                if ep % 2 == 0:
                    save_checkpoint(ticker, target_col, model_type, model, optimizer, ep, best_loss, self.checkpoint_dir)

                # Memory cleanup
                if ep % 3 == 0:
                    self.memory_monitor.cleanup()

            # Save best model
            if best_model_state:
                model.load_state_dict(best_model_state)

            # Final evaluation
            model.eval()
            with torch.no_grad():
                y_pred_tr = model(X_tr_t).numpy().flatten()
                y_pred_va = model(X_va_t).numpy().flatten()

            # ✅ ВИПРАВЛЕНО: Денормалізуємо прогнози (вони були масштабовані під час train)
            y_pred_tr_denorm = y_scaler.inverse_transform(y_pred_tr.reshape(-1, 1)).flatten()
            y_pred_va_denorm = y_scaler.inverse_transform(y_pred_va.reshape(-1, 1)).flatten()

            # Metrics - розраховуємо на денормалізованих даних
            tr_met = compute_metrics(y_tr, y_pred_tr_denorm)
            va_met = compute_metrics(y_va, y_pred_va_denorm)

            # Save model
            m_path = (self.batch_dir / "models" / f"{model_type}_{ticker}_{target_col}.pt")
            self.batch_dir.joinpath("models").mkdir(parents=True, exist_ok=True)

            torch.save({
                'model_state_dict': model.state_dict(),
                'model_type': model_type,
                'input_size': len(available_features),
                'scaler': scaler,
                'y_scaler': y_scaler,  # ✅ Зберігаємо Y scaler
                'features': available_features
            }, m_path)

            # Save Y scaler (для Stage 5)
            import joblib
            scaler_path = self.batch_dir / f"scaler_{ticker}_{target_col}.pkl"
            joblib.dump(y_scaler, scaler_path)
            print(f"✅ Y scaler збережено: {scaler_path}")

            result = {
                "trained": True,
                "model_path": str(m_path),
                "scaler_path": str(scaler_path),
                "mse": best_loss,
                "best_loss": best_loss,
                "train_metrics": tr_met,
                "test_metrics": va_met,
                "feature_count": len(available_features)
            }

            print(f"✅ OK (MSE: {best_loss:.5f}, R²: {va_met['r2']:.3f})")
            return result

        except Exception as e:
            print(f"Помилка: {str(e)[:100]}")
            return {"error": str(e), "trained": False}

# ==============================================================================
# 9. RESULTS AGGREGATOR MODULE
# ==============================================================================

class ResultsAggregator:
    """Агрегація та збереження результатів"""

    def __init__(self, batch_dir, batch_name):
        self.batch_dir = batch_dir
        self.batch_name = batch_name

    def aggregate_results(self):
        """Агрегація результатів з окремих файлів тікерів"""
        print("\n" + "="*80)
        print("📊 АГРЕГАЦІЯ РЕЗУЛЬТАТІВ")
        print("="*80)
        print(f"📁 Шукаємо файли в: {self.batch_dir}")

        results_files = list(self.batch_dir.glob("colab_results_*.json"))
        print(f"📁 Знайдено {len(results_files)} файлів результатів")

        if not results_files:
            print("⚠️ Немає файлів результатів для агрегації")
            return None

        summary = {
            'timestamp': datetime.now().isoformat(),
            'batch_name': self.batch_name,
            'total_tickers': len(results_files),
            'ticker_results': {},
            'models_metadata': {}
        }

        for result_file in results_files:
            try:
                with open(result_file, 'r') as f:
                    ticker_data = json.load(f)

                ticker = ticker_data.get('ticker')
                if not ticker:
                    print(f"⚠️ Пропущено {result_file.name}: немає ticker")
                    continue

                summary['ticker_results'][ticker] = {
                    'total_trained': ticker_data.get('total_trained', 0),
                    'total_failed': ticker_data.get('total_failed', 0),
                    'timeframes': ticker_data.get('timeframes', {})
                }

                # Extract models metadata
                for tf, tf_data in ticker_data.get('timeframes', {}).items():
                    for target_col, target_data in tf_data.get('results', {}).items():
                        for model_name, model_data in target_data.get('models', {}).items():
                            if model_data.get('trained', False):
                                model_key = f"{ticker}_{target_col}_{model_name}"
                                summary['models_metadata'][model_key] = {
                                    'ticker': ticker,
                                    'target': target_col,
                                    'model_type': model_name,
                                    'mse': model_data.get('mse', 0.0),
                                    'best_loss': model_data.get('best_loss', 0.0),
                                    'model_path': model_data.get('model_path', ''),
                                    'feature_count': model_data.get('feature_count', 0),
                                    'train_metrics': model_data.get('train_metrics', {}),
                                    'test_metrics': model_data.get('test_metrics', {})
                                }

                print(f"✅ {ticker}: {ticker_data.get('total_trained', 0)} trained, {ticker_data.get('total_failed', 0)} failed")

            except Exception as e:
                print(f"❌ Помилка обробки {result_file.name}: {e}")

        # Save summary
        summary_path = self.batch_dir / "colab_results_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\n✅ Агреговано результати: {len(summary['ticker_results'])} тікерів, {len(summary['models_metadata'])} моделей")
        print(f"💾 Збережено: {summary_path}")

        return summary

# ==============================================================================
# 10. MAIN COLAB CONTROLLER - ORCHESTRATOR
# ==============================================================================

class ColabTrainingController:
    """Головний контролер тренування в Colab"""

    def __init__(self):
        self.env = ColabEnvironment()
        self.config_loader = None
        self.data_loader = None
        self.feature_selector = None
        self.model_trainer = None
        self.results_aggregator = None
        self.memory_monitor = MemoryMonitor()

    def initialize(self):
        """Ініціалізація всіх компонентів"""
        # ⚠️ Ленивые импорты тяжелых библиотек
        import pandas as pd
        import numpy as np
        
        print("🚀 ІНІЦІАЛІЗАЦІЯ COLAB TRAINING CONTROLLER")

        # Setup environment
        self.env.setup_paths()
        batch_dir, models_dir = self.env.setup_batch_directory()

        # Load configuration
        self.config_loader = RuntimeConfigLoader(self.env.PROJECT_PATH)
        self.config_loader.load_runtime_params(batch_dir)

        # Initialize components
        self.data_loader = ColabDataLoader(batch_dir, self.config_loader)
        self.feature_selector = ColabFeatureSelector(self.env.PROJECT_PATH)
        self.model_trainer = HeavyModelTrainer(batch_dir, self.memory_monitor, self.config_loader)
        self.results_aggregator = ResultsAggregator(batch_dir, self.env.BATCH_NAME)

        return batch_dir

    def run_training_pipeline(self):
        """Запуск повного пайплайну тренування"""
        # ⚠️ Ленивые импорты
        import pandas as pd
        import numpy as np
        
        try:
            # Load and validate data
            features_df, targets_df = self.data_loader.load_data()

            # Check cache
            if self.data_loader.check_cache():
                print("📋 Використовуємо кешовані результати")
                return

            # Load model configurations
            from src.config.unified_config_manager import UnifiedConfigManager
            config_manager = UnifiedConfigManager(config_dir=str(Path(self.env.PROJECT_PATH) / "src/config"))
            models_config = config_manager.get('models', {})

            cat = models_config.get('categories', {})
            heavy_models = cat.get('heavy', ['mlp', 'cnn', 'lstm', 'gru', 'transformer', 'tabnet', 'autoencoder'])

            print(f"\n📊 Важкі моделі для тренування: {heavy_models}")

            # Filter tickers and targets
            tickers = [t for t in targets_df['ticker'].unique() if t]
            if self.config_loader.TEST_TICKER:
                if self.config_loader.TEST_TICKER in tickers:
                    tickers = [self.config_loader.TEST_TICKER]
                    print(f"🧪 Фільтровано тікери: {tickers}")

            if not tickers:
                raise ValueError("❌ Немає тікерів для обробки!")

            # Main training loop
            for ticker in tickers:
                print(f"\n{'='*80}")
                print(f"🎯 ОБРОБКА ТІКЕРА: {ticker}")
                print(f"{'='*80}")

                ticker_file = self.data_loader.batch_dir / f"colab_results_{ticker}.json"

                # Load existing results for resume
                ticker_json = self._load_ticker_results(ticker_file, ticker)

                # Process ticker data
                t_feat = features_df[features_df['ticker'] == ticker]
                t_targ = targets_df[targets_df['ticker'] == ticker]

                if t_feat.empty or t_targ.empty:
                    print("  ⚠️ Даних немає, пропускаю.")
                    continue

                # Merge dataframes
                merged = self._merge_ticker_data(t_feat, t_targ)
                if merged.empty:
                    continue

                # Get target columns
                target_cols = [c for c in merged.columns if c.startswith('target_')]
                if self.config_loader.TEST_TARGET:
                    if self.config_loader.TEST_TARGET in target_cols:
                        target_cols = [self.config_loader.TEST_TARGET]

                if not target_cols:
                    continue

                # Process each target
                for target_col in target_cols:
                    tf = target_col.split('_')[-1]
                    print(f"\n  🎯 Таргет: {target_col}")

                    if tf not in ticker_json["timeframes"]:
                        ticker_json["timeframes"][tf] = {"trained": 0, "failed": 0, "results": {}}

                    if target_col not in ticker_json["timeframes"][tf]["results"]:
                        ticker_json["timeframes"][tf]["results"][target_col] = {"models": {}}

                    # Filter data for this target
                    mask = merged[target_col].notna()
                    if mask.sum() < 50:
                        print(f"    ⚠️ Лише {mask.sum()} зразків, занадто мало.")
                        continue

                    # Sample data if too large
                    X_df, y_ser = self._prepare_training_data(merged, mask, target_col)

                    # Train heavy models
                    for m_type in heavy_models:
                        # Skip if already trained
                        m_res_node = ticker_json["timeframes"][tf]["results"][target_col]["models"]
                        if m_type in m_res_node and m_res_node[m_type].get("trained", False):
                            print(f"    ⏭️ {m_type:<14} | Вже навчена. Пропускаємо.")
                            continue

                        # Feature selection
                        selected_features = self._select_features_for_model(X_df, y_ser, ticker, target_col, m_type)

                        if not selected_features:
                            continue

                        # Train model
                        result = self.model_trainer.train_model(
                            ticker, target_col, m_type, X_df, y_ser, selected_features
                        )

                        # Save result
                        result["selected_features"] = selected_features
                        result["feature_count"] = len(selected_features)

                        k = ticker_json["timeframes"][tf]["results"][target_col]
                        if "models" not in k:
                            k["models"] = {}
                        k["models"][m_type] = result

                        if result.get("trained", False):
                            ticker_json["timeframes"][tf]["trained"] += 1
                            ticker_json["total_trained"] += 1
                        else:
                            ticker_json["timeframes"][tf]["failed"] += 1
                            ticker_json["total_failed"] += 1

                # Save ticker results
                self._save_ticker_results(ticker_file, ticker_json)

            # Save cache signature
            self.data_loader.save_cache_signature()

            # Aggregate results
            self.results_aggregator.aggregate_results()

            # Save memory log
            memory_log_path = self.data_loader.batch_dir / "memory_log.json"
            self.memory_monitor.save_log(memory_log_path)

            print("\n" + "="*80)
            print("🎉 ВСІ ВАЖКІ МОДЕЛІ СФОРМОВАНІ!")
            print(f"👉 Можна вантажити colab_results_summary.json назад локально.")
            print("="*80)

        except Exception as e:
            print(f"❌ КРИТИЧНА ПОМИЛКА В ПАЙПЛАЙНІ: {e}")
            import traceback
            traceback.print_exc()

    def _load_ticker_results(self, ticker_file, ticker):
        """Завантаження існуючих результатів тікера"""
        ticker_json = {
            "ticker": ticker,
            "timestamp": datetime.now().isoformat(),
            "total_trained": 0,
            "total_failed": 0,
            "timeframes": {}
        }

        if ticker_file.exists():
            try:
                with open(ticker_file, 'r') as f:
                    loaded_json = json.load(f)
                    if loaded_json.get('ticker') == ticker:
                        ticker_json = loaded_json
                        print(f"\n📂 Знайдено існуючі результати для {ticker}. Вмикаємо гранулярне доучування.")
            except Exception as e:
                print(f"⚠️ Не вдалося завантажити існуючий файл для {ticker}: {e}")

        return ticker_json

    def _merge_ticker_data(self, t_feat, t_targ):
        """Merge features and targets for ticker"""
        import pandas as pd
        
        common_cols = ['ticker']
        if 'interval' in t_feat.columns and 'interval' in t_targ.columns:
            common_cols.append('interval')
        if 'datetime' in t_feat.columns and 'datetime' in t_targ.columns:
            common_cols.append('datetime')

        merged = pd.merge(t_feat, t_targ, on=common_cols, how='inner')
        print(f"  ✅ Merged: {merged.shape} (inner join з {len(common_cols)} ключами)")
        return merged

    def _prepare_training_data(self, merged, mask, target_col):
        """Підготовка даних для тренування"""
        import numpy as np
        import pandas as pd
        
        print(f"    📊 Data size: {mask.sum()} samples, {len(merged.columns)} columns")

        max_samples_for_selection = 50000
        if mask.sum() > max_samples_for_selection:
            print(f"    ⚠️ Дані занадто великі ({mask.sum()} > {max_samples_for_selection}), використовуємо вибірку")
            sample_idx = np.random.choice(np.where(mask)[0], size=max_samples_for_selection, replace=False)
            X_sample = merged.iloc[sample_idx].drop(columns=[c for c in ['ticker', 'timeframe', 'interval', 'datetime', 'date', 'hash', 'symbol'] if c in merged.columns] + [target_col], errors='ignore')
            y_sample = merged.iloc[sample_idx][target_col].fillna(0)
        else:
            X_sample = merged.loc[mask].drop(columns=[c for c in ['ticker', 'timeframe', 'interval', 'datetime', 'date', 'hash', 'symbol'] if c in merged.columns] + [target_col], errors='ignore')
            y_sample = merged.loc[mask, target_col].fillna(0)

        # Data preprocessing
        X_df = X_sample
        y_ser = y_sample

        X_df = X_df.select_dtypes(exclude=['datetime64', 'datetime', 'datetimetz'])
        for b_col in X_df.select_dtypes(include=['bool']).columns:
            X_df[b_col] = X_df[b_col].astype(float)
        for col in X_df.select_dtypes(include=['object']).columns:
            if X_df[col].nunique() < 20:
                dummies = pd.get_dummies(X_df[col], prefix=col, drop_first=True)
                X_df = pd.concat([X_df.drop(columns=[col]), dummies], axis=1)
            else:
                X_df = X_df.drop(columns=[col], errors='ignore')
        X_df = X_df.apply(pd.to_numeric, errors='coerce').fillna(0).replace([np.inf, -np.inf], 0)

        return X_df, y_ser

    def _select_features_for_model(self, X_df, y_ser, ticker, target_col, m_type):
        """Вибір фіч для конкретної моделі"""
        try:
            from src.config.unified_config_manager import UnifiedConfigManager
            config_manager = UnifiedConfigManager(config_dir=str(Path(self.env.PROJECT_PATH) / "src/config"))
            models_config = config_manager.get('models', {})
            per_mod = models_config.get('per_model', {})
            max_feats = per_mod.get(m_type, {}).get('max_features', 100)

            selected_features = self.feature_selector.select_features(
                X=X_df,
                y=y_ser,
                context_id=f"{ticker}_{target_col}_{m_type}",
                is_classification=False,
                max_features=max_feats
            )

            if len(selected_features) == 0:
                print(f"      ⚠️ SmartSelector повернув 0 фіч, використовуємо fallback")
                correlations = X_df.apply(lambda x: x.corr(y_ser, method='spearman')).abs().sort_values(ascending=False)
                selected_features = correlations.head(max_feats).index.tolist()

            # Save selected features
            feat_file_name = f"selected_features_{m_type}_{ticker}_{target_col}.json"
            feat_path = self.data_loader.batch_dir / feat_file_name
            with open(feat_path, 'w') as f:
                json.dump({
                    "ticker": ticker,
                    "target": target_col,
                    "model_type": m_type,
                    "selected_features": selected_features,
                    "feature_count": len(selected_features),
                    "max_features": max_feats,
                    "timestamp": datetime.now().isoformat()
                }, f, indent=2)

            return selected_features

        except Exception as e:
            print(f"    ❌ Помилка вибору фіч для {m_type}: {str(e)[:100]}")
            return None

    def _save_ticker_results(self, ticker_file, ticker_json):
        """Збереження результатів тікера"""
        print(f"\n💾 Збереження результатів для {ticker_json['ticker']}...")
        print(f"   📁 Шлях: {ticker_file}")
        print(f"   📊 Дані: {ticker_json.get('total_trained', 0)} trained, {ticker_json.get('total_failed', 0)} failed")

        try:
            with open(ticker_file, 'w') as f:
                json.dump(ticker_json, f, indent=2)
            print(f"✅ Результати {ticker_json['ticker']} збережено: {ticker_file.name}")
        except Exception as e:
            print(f"❌ ПОМИЛКА при збереженні {ticker_file}: {e}")

# ==============================================================================
# MAIN EXECUTION - REFACTORED
# ==============================================================================

if __name__ == "__main__":
    print("🔥 COLAB TRAINING CONTROLLER - REFACTORED VERSION")
    print("="*80)

    # Initialize controller
    controller = ColabTrainingController()

    # Run initialization
    batch_dir = controller.initialize()

    # Run training pipeline
    controller.run_training_pipeline()

    print("\n✅ COLAB TRAINING COMPLETED!")