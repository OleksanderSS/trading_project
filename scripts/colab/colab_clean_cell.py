# ==============================================================================
# COLAB TRAINING CONTROLLER - CLEAN VERSION
# ==============================================================================

import os
import sys
import json
import warnings
from datetime import datetime
from pathlib import Path

from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger(__name__)

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# ML Audit tools
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# Colab-specific imports
try:
    from google.colab import drive
    COLAB_AVAILABLE = True
except ImportError:
    COLAB_AVAILABLE = False

# Constants
CONTENT_DRIVE_PATH = "/content/drive/MyDrive/trading_project"
RUNTIME_PARAMS_FILE = "runtime_params.json"

# ==============================================================================
# DTYPE FIX FUNCTIONS
# ==============================================================================

def _fix_base_neural_model():
    """Виправити BaseNeuralModel dtype"""
    from src.models.neural.base_neural import BaseNeuralModel

    original_train = BaseNeuralModel.train
    def fixed_train(self, x, y, epochs=50, batch_size=32, validation_split=0.2, **kwargs):
        # Перетворення в numpy з правильними типами даних
        x_np = x.values if isinstance(x, pd.DataFrame) else np.asarray(x)
        y_np = y.values if isinstance(y, (pd.Series, pd.DataFrame)) else np.asarray(y)

        # Перетворення в числові типи для Keras
        x_np = x_np.astype(np.float32)
        y_np = y_np.astype(np.float32)

        # Заміна NaN/inf для стабільності
        x_np = np.nan_to_num(x_np, nan=0.0, posinf=0.0, neginf=0.0)
        y_np = np.nan_to_num(y_np, nan=0.0, posinf=0.0, neginf=0.0)

        # Виклик оригінального методу з виправленими даними
        return original_train(self, x_np, y_np, epochs, batch_size, validation_split, **kwargs)

    setattr(BaseNeuralModel, 'train', fixed_train)  # type: ignore
    print("✅ BaseNeuralModel виправлено")

def _fix_cnn_model():
    """Виправити CNNModel dtype"""
    from src.models.neural.cnn_model import CNNModel

    original_cnn_train = CNNModel.train
    def fixed_cnn_train(self, X, y, **kwargs):
        # Перетворення в numpy з правильними типами
        x_array = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
        y_array = y.values if isinstance(y, (pd.Series, pd.DataFrame)) else np.asarray(y)

        # Перетворення в числові типи для Keras
        x_array = x_array.astype(np.float32)
        y_array = y_array.astype(np.float32)

        # Виклик оригінального методу з виправленими даними
        return original_cnn_train(self, x_array, y_array, **kwargs)

    setattr(CNNModel, 'train', fixed_cnn_train)  # type: ignore
    print("✅ CNNModel виправлено")

def _fix_lstm_model():
    """Виправити LSTMModel dtype"""
    from src.models.neural.lstm_model import LSTMModel

    original_lstm_train = LSTMModel.train
    def fixed_lstm_train(self, X, y, **kwargs):
        # Reshape data if it's 2D
        if len(X.shape) == 2:
            X = np.reshape(X, (X.shape[0], X.shape[1], 1))

        # Перетворення в числові типи для Keras
        X = X.astype(np.float32)
        y = y.astype(np.float32)

        return original_lstm_train(self, X, y, **kwargs)

    setattr(LSTMModel, 'train', fixed_lstm_train)  # type: ignore
    print("✅ LSTMModel виправлено")

def _fix_gru_model():
    """Виправити GRUModel dtype"""
    from src.models.neural.gru_model import GRUModel

    original_gru_train = GRUModel.train
    def fixed_gru_train(self, X, y, **kwargs):
        # Reshape data if it's 2D
        if len(X.shape) == 2:
            X = np.reshape(X, (X.shape[0], X.shape[1], 1))

        # Перетворення в числові типи для Keras
        X = X.astype(np.float32)
        y = y.astype(np.float32)

        return original_gru_train(self, X, y, **kwargs)

    setattr(GRUModel, 'train', fixed_gru_train)  # type: ignore
    print("✅ GRUModel виправлено")

def _fix_transformer_model():
    """Виправити TransformerModel dtype"""
    from src.models.neural.transformer_model import TransformerModel

    original_transformer_fit = TransformerModel.fit
    def fixed_transformer_fit(self, X, y, seq_len=10, epochs=20, batch_size=32):
        # Convert to numpy
        if hasattr(X, 'values'):
            X = X.values
        if hasattr(y, 'values'):
            y = y.values

        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

        # Перетворення в числові типи для Keras
        X = X.astype(np.float32)
        y = y.astype(np.float32)

        # Виклик оригінального методу з виправленими даними
        return original_transformer_fit(self, X, y, seq_len, epochs, batch_size)

    setattr(TransformerModel, 'fit', fixed_transformer_fit)  # type: ignore
    print("✅ TransformerModel виправлено")

def fix_neural_models_dtype():
    """Виправити проблему з dtype в нейронних моделях"""

    print("🔧 ВИПРАВЛЕННЯ DTYPE В НЕЙРОННИХ МОДЕЛЯХ...")

    # Виправити кожну модель окремо
    try:
        _fix_base_neural_model()
    except Exception as e:
        print(f"❌ Помилка виправлення BaseNeuralModel: {e}")

    try:
        _fix_cnn_model()
    except Exception as e:
        print(f"❌ Помилка виправлення CNNModel: {e}")

    try:
        _fix_lstm_model()
    except Exception as e:
        print(f"❌ Помилка виправлення LSTMModel: {e}")

    try:
        _fix_gru_model()
    except Exception as e:
        print(f"❌ Помилка виправлення GRUModel: {e}")

    try:
        _fix_transformer_model()
    except Exception as e:
        print(f"❌ Помилка виправлення TransformerModel: {e}")

    print("🎯 ВСІ НЕЙРОННІ МОДЕЛІ ВИПРАВЛЕНО!")

# ==============================================================================
# 1. CONFIGURATION LOADER
# ==============================================================================

class ConfigLoader:
    """Завантаження конфігурації для Colab"""

    def __init__(self, project_path: str | Path):
        self.project_path = Path(project_path)
        self.runtime_params: dict = {}
        self.test_mode: dict = {}
        self.TEST_TICKER: str | None = None
        self.TEST_TARGET: str | None = None
        self.REDUCED_EPOCHS: int = 1
        self.MAX_ITERATIONS: int = 100
        self._load_runtime_params()

    def _load_runtime_params(self):
        """Завантаження runtime параметрів"""
        # Search in multiple locations
        possible_paths = [
            self.project_path / "data" / "colab" / "accumulated" / "main_database" / RUNTIME_PARAMS_FILE,
            self.project_path / "src" / "config" / RUNTIME_PARAMS_FILE,
            self.project_path / RUNTIME_PARAMS_FILE
        ]

        for runtime_params_path in possible_paths:
            if runtime_params_path.exists():
                with open(runtime_params_path) as f:
                    self.runtime_params = json.load(f)

                # Extract parameters
                self.TEST_TICKER = self.runtime_params.get('test_ticker')
                self.TEST_TARGET = self.runtime_params.get('test_target')
                self.REDUCED_EPOCHS = self.runtime_params.get('epochs', 1)  # Default to 1 for fast testing
                self.MAX_ITERATIONS = self.runtime_params.get('max_iterations', 1)

                # Extract full parameters
                self.TIMEFRAMES = self.runtime_params.get('timeframes', ['15m', '1h', '1d'])
                self.TICKERS = self.runtime_params.get('tickers', 'all')

                self._print_loaded_params()
                break
        else:
            print("⚠️ runtime_params.json не знайдено, використовуємо параметри за замовчуванням")
            # Full mode - не обмежуємо епохи/ітерації
            self.REDUCED_EPOCHS = 100  # Default for full mode
            self.MAX_ITERATIONS = 100  # Default for full mode
            # Full mode defaults
            self.TIMEFRAMES = ['15m', '1h', '1d']
            self.TICKERS = 'all'

    def _print_loaded_params(self):
        """Вивід завантажених параметрів"""
        print("\n" + "="*80)
        print("📥 ПАРАМЕТРИ ЗАВАНТАЖЕНО З runtime_params.json")
        print("="*80)
        print(f"  Режим: {self.runtime_params.get('mode', 'full')}")
        print(f"  Тікери: {self.TEST_TICKER if self.TEST_TICKER else 'всі з конфігу'}")
        print(f"  Таргети: {self.TEST_TARGET if self.TEST_TARGET else 'всі з конфігу'}")
        print(f"  Таймфрейми: {self.TIMEFRAMES}")
        print(f"  Тікери: {self.TICKERS}")
        print(f"  Епохи: {self.REDUCED_EPOCHS}")
        print(f"  Ітерації: {self.MAX_ITERATIONS}")
        print("="*80 + "\n")

# ==============================================================================
# 2. PATH MANAGER
# ==============================================================================

class PathManager:
    """Управління шляхами для Colab"""

    def __init__(self):
        self.PROJECT_PATH: Path = Path(".")
        self.SRC_PATH: str = "."
        self.batch_dir: Path = Path("./data/batches")
        self.setup_paths()

    def setup_paths(self):
        """Налаштування шляхів для Colab або локального середовища"""
        if os.path.exists(CONTENT_DRIVE_PATH):
            self.PROJECT_PATH = CONTENT_DRIVE_PATH
            print("✅ Google Drive вже підключено")
        else:
            if COLAB_AVAILABLE:
                try:
                    drive.mount('/content/drive', force_remount=False)
                    self.PROJECT_PATH = CONTENT_DRIVE_PATH
                    print("✅ Google Drive підключено")
                except Exception as e:
                    self.PROJECT_PATH = str(Path.cwd())
                    print(f"⚠️ Не вдалося підключити Google Drive: {e}")
            else:
                self.PROJECT_PATH = str(Path.cwd())
                print(f"⚠️ Працюємо локально з {self.PROJECT_PATH}")

        self.SRC_PATH = str(Path(self.PROJECT_PATH) / "src")
        
        # Ensure project path is a Path object
        self.PROJECT_PATH = Path(self.PROJECT_PATH)

        # Update sys.path
        for p in [self.SRC_PATH, str(self.PROJECT_PATH)]:
            if p not in sys.path:
                sys.path.insert(0, p)

        print(f"📁 PROJECT_PATH: {self.PROJECT_PATH}")
        print(f"📁 SRC_PATH: {self.SRC_PATH}")

        # Set batch directory
        self.batch_dir = Path(self.PROJECT_PATH) / "data" / "colab" / "accumulated" / "main_database"

# ==============================================================================
# 3. MEMORY MONITOR
# ==============================================================================

class MemoryMonitor:
    """Моніторинг пам'яті в Colab"""

    def check_memory(self, context=""):
        """Перевірка використання пам'яті"""
        try:
            import psutil
            memory = psutil.virtual_memory()
            used_percent = memory.percent

            if used_percent > 90:
                return 'critical'
            elif used_percent > 80:
                return 'warning'
            else:
                return 'ok'
        except ImportError:
            return 'unknown'

    def get_memory_usage(self):
        """Отримати відсоток використання пам'яті"""
        try:
            import psutil
            return psutil.virtual_memory().percent
        except ImportError:
            return 0

# ==============================================================================
# 4. FEATURE SELECTOR
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
            # Clear conflicting paths
            paths_to_remove = [p for p in sys.path if p.endswith('src') or p.endswith('src/')]
            for path in paths_to_remove:
                sys.path.remove(path)

            # Add correct paths
            src_path = self.project_path / "src"
            project_path_str = str(self.project_path)
            src_path_str = str(src_path)

            sys.path.insert(0, project_path_str)
            sys.path.insert(0, src_path_str)

            # Import and initialize
            from src.features.selection.smart_selector import SmartFeatureSelector

            cache_path = self.project_path / "data" / "cache" / "selected_features.json"
            cache_path.parent.mkdir(parents=True, exist_ok=True)

            self.feature_selector = SmartFeatureSelector(storage_path=str(cache_path))
            print("✅ SmartFeatureSelector ініціалізовано")

        except Exception as e:
            print(f"❌ Помилка при ініціалізації SmartFeatureSelector: {e}")
            raise

    def select_features(self, X, y, context_id, is_classification=False, max_features=None):
        """Вибір фіч для моделі"""
        if self.feature_selector is None:
            print("⚠️ SmartFeatureSelector не ініціалізовано, повертаємо всі фічі")
            return X.columns.tolist()
            
        return self.feature_selector.select(
            features_df=X,
            target_series=y,
            context_id=context_id,
            is_classification=is_classification,
            market_regime="normal",
            force_recalculate=False,
            max_features=max_features
        )

# ==============================================================================
# 5. DATA LOADER
# ==============================================================================

class ColabDataLoader:
    """Завантаження та підготовка даних для Colab"""

    def __init__(self, batch_dir, config_loader):
        self.batch_dir = Path(batch_dir)
        self.config_loader = config_loader
        self.features_df = None
        self.targets_df = None

    def load_data(self):
        """Завантаження features та targets"""
        features_path = self.batch_dir / "features.parquet"
        targets_path = self.batch_dir / "targets.parquet"

        if not features_path.exists() or not targets_path.exists():
            raise FileNotFoundError(f"Файли відсутні в {self.batch_dir}")

        self.features_df = pd.read_parquet(features_path)
        self.targets_df = pd.read_parquet(targets_path)

        print(f"✅ Завантажено дані з {self.batch_dir}")
        print(f"   Features: {self.features_df.shape}")
        print(f"   Targets: {self.targets_df.shape}")

        # Remove target columns from features
        target_cols_in_features = [c for c in self.features_df.columns
                                 if c in self.targets_df.columns and c not in ['ticker', 'datetime', 'interval']]
        if target_cols_in_features:
            print(f"⚠️ Видаляємо {len(target_cols_in_features)} target колонок з features")
            self.features_df = self.features_df.drop(columns=target_cols_in_features)

        # Normalize timezones
        if 'datetime' in self.features_df.columns:
            self.features_df['datetime'] = pd.to_datetime(self.features_df['datetime']).dt.tz_localize(None)
        if 'datetime' in self.targets_df.columns:
            self.targets_df['datetime'] = pd.to_datetime(self.targets_df['datetime']).dt.tz_localize(None)

        print("✅ Timezone нормалізовано")
        return self.features_df, self.targets_df

# ==============================================================================
# 6. MAIN CONTROLLER
# ==============================================================================

class ColabTrainingController:
    """Основний контролер для тренування в Colab"""

    def __init__(self):
        self.logger = ProjectLogger.get_logger("ColabTrainingController")
        self.path_manager = PathManager()
        self.config_loader = ConfigLoader(self.path_manager.PROJECT_PATH or Path("."))
        self.memory_monitor = MemoryMonitor()
        self.data_loader = ColabDataLoader(self.path_manager.batch_dir or Path("."), self.config_loader)
        self.feature_selector = ColabFeatureSelector(self.path_manager.PROJECT_PATH or Path("."))
        self.results = {
            'ticker_results': {},
            'models_metadata': {},
            'timestamp': datetime.now().isoformat(),
            'batch_name': self.path_manager.batch_dir.name if self.path_manager.batch_dir else "unknown"
        }

    def initialize(self):
        """Ініціалізація контролера"""
        print("🚀 Ініціалізація ColabTrainingController...")
        self.path_manager.setup_paths()

        # Авто-встановлення відсутніх бібліотек у Colab
        if COLAB_AVAILABLE:
            try:
                import pytorch_tabnet
            except ImportError:
                print("📦 Встановлення pytorch-tabnet...")
                import subprocess
                subprocess.check_call([sys.executable, "-m", "pip", "install", "pytorch-tabnet"])
                print("✅ pytorch-tabnet встановлено")

        # Виправити проблеми з типами даних
        fix_neural_models_dtype()

        # Очистити кеші перед початком
        self.clear_caches()

    def clear_caches(self):
        """Очищення кешів та пам'яті"""
        print("🧹 Очищення кешів...")
        
        # Очистити кеш SmartFeatureSelector
        try:
            from src.features.selection.smart_selector import SmartFeatureSelector
            if hasattr(SmartFeatureSelector, '_feature_cache'):
                setattr(SmartFeatureSelector, '_feature_cache', {})
            if hasattr(SmartFeatureSelector, '_model_cache'):
                setattr(SmartFeatureSelector, '_model_cache', {})
        except Exception:
            logger.warning("Failed to clear cache or session", exc_info=True)

        # Очистити Keras сесію
        try:
            import tensorflow as tf
            tf.keras.backend.clear_session()
        except Exception:
            logger.warning("Failed to clear cache or session", exc_info=True)
        
        # GC
        import gc
        gc.collect()

    def run_training_pipeline(self):
        """Запуск повного пайплайну тренування"""
        try:
            # Load data
            features_df, targets_df = self.data_loader.load_data()

            # Setup MLflow experiment
            if MLFLOW_AVAILABLE:
                mlflow.set_experiment(f"Trading_Audit_{datetime.now().strftime('%Y%m%d')}")
                print("📝 MLflow експеримент налаштовано")

            # Filter tickers
            tickers = self._filter_tickers(targets_df)

            # Get heavy models
            heavy_models = ['mlp', 'cnn', 'lstm', 'gru', 'transformer', 'tabnet', 'autoencoder']
            print(f"📊 Важкі моделі для тренування: {heavy_models}")

            # Process each ticker
            for ticker in tickers:
                self._process_ticker(ticker, features_df, targets_df, heavy_models)

            # Save results summary
            self._save_results_summary()

            print("\n✅ Тренування завершено! Звіт збережено.")

        except Exception as e:
            print(f"❌ Помилка в пайплайні: {e}")
            raise

    def _filter_tickers(self, targets_df):
        """Фільтрація тікерів"""
        tickers = [t for t in targets_df['ticker'].unique() if t]
        if self.config_loader.TEST_TICKER:
            if self.config_loader.TEST_TICKER in tickers:
                tickers = [self.config_loader.TEST_TICKER]
                print(f"🧪 Фільтровано тікери: {tickers}")

        if not tickers:
            raise ValueError("❌ Немає тікерів для обробки!")

        return tickers

    def _process_ticker(self, ticker, features_df, targets_df, heavy_models):
        """Обробка одного тікера"""
        print(f"\n{'='*80}")
        print(f"🎯 ОБРОБКА ТІКЕРА: {ticker}")
        print(f"{'='*80}")

        # Get ticker data
        t_feat = features_df[features_df['ticker'] == ticker]
        t_targ = targets_df[targets_df['ticker'] == ticker]

        if t_feat.empty or t_targ.empty:
            print("  ⚠️ Даних немає, пропускаю.")
            return

        # Merge data
        common_cols = ['ticker']
        if 'datetime' in t_feat.columns and 'datetime' in t_targ.columns:
            common_cols.append('datetime')

        merged = pd.merge(t_feat, t_targ, on=common_cols, how='inner', validate='one_to_one')
        print(f"  ✅ Merged: {merged.shape}")

        # Get target columns
        target_cols = [c for c in merged.columns if c.startswith('target_')]
        if self.config_loader.TEST_TARGET:
            if self.config_loader.TEST_TARGET in target_cols:
                target_cols = [self.config_loader.TEST_TARGET]

        # Process each target
        for target_col in target_cols:
            self._process_target(ticker, target_col, merged, heavy_models)

    def _process_target(self, ticker, target_col, merged, heavy_models):
        """Обробка одного цільового стовпця"""
        print(f"\n  🎯 Таргет: {target_col}")
        
        # Initialize results structure for this ticker/target
        if ticker not in self.results['ticker_results']:
            self.results['ticker_results'][ticker] = {'timeframes': {'all': {'results': {}}}}
        
        if target_col not in self.results['ticker_results'][ticker]['timeframes']['all']['results']:
            self.results['ticker_results'][ticker]['timeframes']['all']['results'][target_col] = {'models': {}}

        # Filter data
        mask = merged[target_col].notna()
        if mask.sum() < 50:
            print(f"    ⚠️ Лише {mask.sum()} зразків, занадто мало.")
            return

        print(f"    📊 Data size: {mask.sum()} samples, {len(merged.columns)} columns")

        # Prepare training data
        x_df = merged.loc[mask].drop(columns=['ticker', 'datetime'] + [c for c in merged.columns if c.startswith('target_')], errors='ignore')
        y_ser = merged.loc[mask, target_col]

        # Process data types
        # Process data types - CRITICAL FIX FOR DTYPE OBJECT
        x_df = x_df.select_dtypes(exclude=['datetime64', 'datetime', 'object'])
        x_df = x_df.apply(pd.to_numeric, errors='coerce').fillna(0).astype(np.float32)
        x_df = x_df.replace([np.inf, -np.inf], 0)

        y_ser = pd.to_numeric(y_ser, errors='coerce').fillna(0).astype(np.float32)

        # Scaling - CRITICAL FOR NEURAL NETWORKS
        try:
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            x_scaled = pd.DataFrame(
                scaler.fit_transform(x_df),
                columns=x_df.columns,
                index=x_df.index
            )
            x_df = x_scaled
            print("    ⚖️ Дані масштабовано (StandardScaler)")
        except Exception as e:
            print(f"    ⚠️ Помилка масштабування: {e}")

        # Train models
        for model_type in heavy_models:
            self._train_model(ticker, target_col, model_type, x_df, y_ser)

    def _train_model(self, ticker, target_col, model_type, x_df, y_ser):
        """Тренування однієї моделі"""
        # Check if model already exists to skip re-training
        ext = ".keras" if model_type in ['cnn', 'lstm', 'gru', 'transformer', 'autoencoder'] else ".pkl"
        if model_type == 'tabnet': ext = ".zip"
        
        model_filename = f"model_{ticker}_{target_col}_{model_type}{ext}"
        model_path = self.path_manager.batch_dir / model_filename
        
        if model_path.exists():
            print(f"    ⏭️  {model_type:<14} | Вже існує, пропускаю.")
            
            # Record result for skipped model
            try:
                max_features = self._get_model_max_features(model_type)
                selected_features = self.feature_selector.select_features(
                    X=x_df, y=y_ser, context_id=f"{ticker}_{target_col}_{model_type}",
                    is_classification=False, max_features=max_features
                )
            except Exception as e:
                self.logger.error("Failed to select features, using all columns", exc_info=True)
                selected_features = list(x_df.columns)

            model_result = {
                'status': 'success',
                'model_path': model_filename,
                'metrics': {'info': 'already_exists'},
                'selected_features': selected_features
            }
            
            if ticker not in self.results['ticker_results']:
                self.results['ticker_results'][ticker] = {'timeframes': {'all': {'results': {}}}}
            if target_col not in self.results['ticker_results'][ticker]['timeframes']['all']['results']:
                self.results['ticker_results'][ticker]['timeframes']['all']['results'][target_col] = {'models': {}}
            
            self.results['ticker_results'][ticker]['timeframes']['all']['results'][target_col]['models'][model_type] = model_result
            
            # Add to models_metadata
            meta_key = f"{ticker}_{target_col}_{model_type}"
            self.results['models_metadata'][meta_key] = {
                'ticker': ticker,
                'target': target_col,
                'model_type': model_type,
                'path': model_filename,
                'metrics': {'info': 'already_exists'},
                'selected_features': selected_features
            }
            return

        print(f"    🔍 {model_type:<14} | ", end="")

        try:
            # Feature selection
            max_features = self._get_model_max_features(model_type)
            selected_features = self.feature_selector.select_features(
                X=x_df,
                y=y_ser,
                context_id=f"{ticker}_{target_col}_{model_type}",
                is_classification=False,
                max_features=max_features
            )

            if len(selected_features) == 0:
                print("❌ 0 фіч вибрано")
                return

            print(f"✅ OK ({len(selected_features)} фіч)")

            # Train the model
            metrics = self._train_model_with_features(ticker, target_col, model_type, x_df, y_ser, selected_features)
            
            # Record result
            ext = ".keras" if model_type in ['cnn', 'lstm', 'gru', 'transformer', 'autoencoder'] else ".pkl"
            if model_type == 'tabnet': ext = ".zip"
            model_filename = f"model_{ticker}_{target_col}_{model_type}{ext}"
            
            model_result = {
                'status': 'success',
                'model_path': model_filename,
                'metrics': metrics,
                'selected_features': selected_features
            }
            
            self.results['ticker_results'][ticker]['timeframes']['all']['results'][target_col]['models'][model_type] = model_result
            
            # Add to models_metadata for easier access
            meta_key = f"{ticker}_{target_col}_{model_type}"
            self.results['models_metadata'][meta_key] = {
                'ticker': ticker,
                'target': target_col,
                'model_type': model_type,
                'path': model_filename,
                'metrics': metrics,
                'selected_features': selected_features
            }

        except Exception as e:
            print(f"❌ Помилка: {str(e)[:100]}")
            # Record error
            if ticker in self.results['ticker_results'] and target_col in self.results['ticker_results'][ticker]['timeframes']['all']['results']:
                self.results['ticker_results'][ticker]['timeframes']['all']['results'][target_col]['models'][model_type] = {
                    'status': 'error',
                    'message': str(e)[:100]
                }

    def _train_model_with_features(self, ticker, target_col, model_type, x_df, y_ser, selected_features):
        """Тренування моделі з вибраними фічами"""
        try:
            # Prepare data with selected features
            x_train = x_df[selected_features]

            # Create model based on type
            if model_type == 'mlp':
                self._train_mlp_model(x_train, y_ser, ticker, target_col)
            elif model_type == 'cnn':
                self._train_cnn_model(x_train, y_ser, ticker, target_col)
            elif model_type == 'lstm':
                self._train_lstm_model(x_train, y_ser, ticker, target_col)
            elif model_type == 'gru':
                self._train_gru_model(x_train, y_ser, ticker, target_col)
            elif model_type == 'transformer':
                self._train_transformer_model(x_train, y_ser, ticker, target_col)
            elif model_type == 'tabnet':
                self._train_tabnet_model(x_train, y_ser, ticker, target_col)
            elif model_type == 'autoencoder':
                self._train_autoencoder_model(x_train, y_ser, ticker, target_col)
            else:
                print(f"⚠️ Невідомий тип моделі: {model_type}")

        except Exception as e:
            print(f"❌ Помилка тренування {model_type}: {str(e)[:100]}")
            return {'error': str(e)[:100]}
            
    def _save_results_summary(self):
        """Збереження звіту про результати та файлів вибраних фіч"""
        if not self.path_manager.batch_dir:
            print("⚠️ Неможливо зберегти звіт: batch_dir не встановлено")
            return

        # 1. Save main results JSON
        results_path = self.path_manager.batch_dir / "colab_results.json"
        try:
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(self.results, f, indent=2)
            print(f"📝 Звіт збережено у {results_path}")
        except Exception as e:
            print(f"❌ Помилка збереження звіту: {e}")

        # 2. Export individual selected_features_*.json for local pipeline compatibility
        print("📤 Експорт файлів вибраних фіч для локального пайплайну...")
        exported_count = 0
        for meta_key, meta in self.results.get('models_metadata', {}).items():
            if 'selected_features' in meta:
                ticker = meta['ticker']
                target = meta['target']
                model = meta['model_type']
                
                # Format required by HybridOrchestrator
                fs_data = {
                    'ticker': ticker,
                    'targets': [target],
                    'model_name': model,
                    'selected_features': meta['selected_features'],
                    'timestamp': datetime.now().isoformat()
                }
                
                fs_filename = f"selected_features_{ticker}_{target}_{model}.json"
                fs_path = self.path_manager.batch_dir / fs_filename
                
                try:
                    with open(fs_path, 'w', encoding='utf-8') as f:
                        json.dump(fs_data, f, indent=2)
                    exported_count += 1
                except Exception as e:
                    self.logger.error(f"Failed to export selected features for {ticker}_{target}_{model}", exc_info=True)
                    pass
        
        if exported_count > 0:
            print(f"✅ Експортовано {exported_count} файлів вибраних фіч")

    def _log_mlflow_run(self, ticker, target, model_type, params, metrics, artifact_path=None):
        """Helper to log results to MLflow"""
        if not MLFLOW_AVAILABLE:
            return

        try:
            with mlflow.start_run(run_name=f"{ticker}_{target}_{model_type}", nested=True):
                mlflow.log_param("ticker", ticker)
                mlflow.log_param("target", target)
                mlflow.log_param("model_type", model_type)

                # Log hyperparameters
                for p_name, p_val in params.items():
                    mlflow.log_param(p_name, p_val)

                # Log metrics
                for m_name, m_val in metrics.items():
                    mlflow.log_metric(m_name, m_val)

                # Log model artifact
                if artifact_path and os.path.exists(artifact_path):
                    mlflow.log_artifact(artifact_path)
        except Exception as e:
            print(f"⚠️ Помилка логування в MLflow: {e}")

    def _train_mlp_model(self, x_train, y_train, ticker, target_col):
        """Тренування MLP моделі"""
        import joblib
        from sklearn.metrics import mean_squared_error
        from sklearn.model_selection import train_test_split
        from sklearn.neural_network import MLPRegressor

        # Split data
        x_train_split, x_val, y_train_split, y_val = train_test_split(
            x_train, y_train, test_size=0.2, random_state=42
        )

        # Create and train model
        model = MLPRegressor(
            hidden_layer_sizes=(128, 64),
            max_iter=self.config_loader.REDUCED_EPOCHS,
            random_state=42,
            verbose=0
        )

        model.fit(x_train_split, y_train_split)

        # Evaluate
        y_pred = model.predict(x_val)
        mse = mean_squared_error(y_val, y_pred)

        # Save model
        model_path = self.path_manager.batch_dir / f"model_{ticker}_{target_col}_mlp.pkl"
        joblib.dump(model, model_path)

        # Log to MLflow
        self._log_mlflow_run(
            ticker, target_col, "mlp",
            params={"hidden_layers": "128,64", "max_iter": 1},
            metrics={"mse": mse},
            artifact_path=str(model_path)
        )

        print(f"    🎯 MLP - MSE: {mse:.6f} - збережено: {model_path.name}")
        return {'mse': float(mse)}

    def _train_cnn_model(self, x_train, y_train, ticker, target_col):
        """Тренування CNN моделі"""
        import tensorflow as tf
        from sklearn.model_selection import train_test_split

        # Split data
        x_train_split, x_val, y_train_split, y_val = train_test_split(
            x_train, y_train, test_size=0.2, random_state=42
        )

        # Ensure numpy float32 and reshape for CNN (add channel dimension)
        x_train_reshaped = x_train_split.values.astype(np.float32).reshape(x_train_split.shape[0], x_train_split.shape[1], 1)
        x_val_reshaped = x_val.values.astype(np.float32).reshape(x_val.shape[0], x_val.shape[1], 1)
        y_train_split = y_train_split.values.astype(np.float32)
        y_val = y_val.values.astype(np.float32)

        # Create CNN model
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(x_train.shape[1], 1)),
            tf.keras.layers.Conv1D(32, 3, activation='relu'),
            tf.keras.layers.MaxPooling1D(2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1)
        ])

        model.compile(optimizer='adam', loss='mse')

        # Train
        epochs = self.config_loader.REDUCED_EPOCHS or 10
        history = model.fit(
            x_train_reshaped, y_train_split,
            epochs=epochs,
            validation_data=(x_val_reshaped, y_val),
            verbose=0
        )

        # Save model
        model_path = self.path_manager.batch_dir / f"model_{ticker}_{target_col}_cnn.keras"
        model.save(model_path)

        # Log to MLflow
        self._log_mlflow_run(
            ticker, target_col, "cnn",
            params={"conv_filters": 32, "kernel_size": 3, "epochs": 1},
            metrics={"loss": history.history['loss'][0]},
            artifact_path=str(model_path)
        )

        print(f"    🎯 CNN - Loss: {history.history['loss'][-1]:.6f} - збережено: {model_path.name}")
        return {'loss': float(history.history['loss'][-1])}

    def _train_lstm_model(self, x_train, y_train, ticker, target_col):
        """Тренування LSTM моделі"""
        import tensorflow as tf
        from sklearn.model_selection import train_test_split

        # Split data
        x_train_split, x_val, y_train_split, y_val = train_test_split(
            x_train, y_train, test_size=0.2, random_state=42
        )

        # Ensure numpy float32 and reshape for LSTM (add timestep dimension)
        x_train_reshaped = x_train_split.values.astype(np.float32).reshape(x_train_split.shape[0], 1, x_train_split.shape[1])
        x_val_reshaped = x_val.values.astype(np.float32).reshape(x_val.shape[0], 1, x_val.shape[1])
        y_train_split = y_train_split.values.astype(np.float32)
        y_val = y_val.values.astype(np.float32)

        # Create LSTM model
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(1, x_train.shape[1])),
            tf.keras.layers.LSTM(64),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1)
        ])

        model.compile(optimizer='adam', loss='mse')

        # Train
        epochs = self.config_loader.REDUCED_EPOCHS or 10
        history = model.fit(
            x_train_reshaped, y_train_split,
            epochs=epochs,
            validation_data=(x_val_reshaped, y_val),
            verbose=0
        )

        # Save model
        model_path = self.path_manager.batch_dir / f"model_{ticker}_{target_col}_lstm.keras"
        model.save(model_path)
        
        # Log to MLflow
        self._log_mlflow_run(
            ticker, target_col, "lstm",
            params={"units": 64, "epochs": epochs},
            metrics={"loss": history.history['loss'][0]},
            artifact_path=str(model_path)
        )

        print(f"    🎯 LSTM - Loss: {history.history['loss'][-1]:.6f} - збережено: {model_path.name}")
        return {'loss': float(history.history['loss'][-1])}

    def _train_gru_model(self, x_train, y_train, ticker, target_col):
        """Тренування GRU моделі"""
        import tensorflow as tf
        from sklearn.model_selection import train_test_split

        # Split data
        x_train_split, x_val, y_train_split, y_val = train_test_split(
            x_train, y_train, test_size=0.2, random_state=42
        )

        # Ensure numpy float32 and reshape for GRU
        x_train_reshaped = x_train_split.values.astype(np.float32).reshape(x_train_split.shape[0], 1, x_train_split.shape[1])
        x_val_reshaped = x_val.values.astype(np.float32).reshape(x_val.shape[0], 1, x_val.shape[1])
        y_train_split = y_train_split.values.astype(np.float32)
        y_val = y_val.values.astype(np.float32)

        # Create GRU model
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(1, x_train.shape[1])),
            tf.keras.layers.GRU(64),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1)
        ])

        model.compile(optimizer='adam', loss='mse')

        # Train
        epochs = self.config_loader.REDUCED_EPOCHS or 10
        history = model.fit(
            x_train_reshaped, y_train_split,
            epochs=epochs,
            validation_data=(x_val_reshaped, y_val),
            verbose=0
        )

        # Save model
        model_path = self.path_manager.batch_dir / f"model_{ticker}_{target_col}_gru.keras"
        model.save(model_path)
        
        # Log to MLflow
        self._log_mlflow_run(
            ticker, target_col, "gru",
            params={"units": 64, "epochs": epochs},
            metrics={"loss": history.history['loss'][0]},
            artifact_path=str(model_path)
        )

        print(f"    🎯 GRU - Loss: {history.history['loss'][-1]:.6f} - збережено: {model_path.name}")
        return {'loss': float(history.history['loss'][-1])}

    def _train_transformer_model(self, x_train, y_train, ticker, target_col):
        """Тренування Transformer моделі"""
        import tensorflow as tf
        from sklearn.model_selection import train_test_split

        # Split data
        x_train_split, x_val, y_train_split, y_val = train_test_split(
            x_train, y_train, test_size=0.2, random_state=42
        )

        # Ensure numpy float32 and reshape for Transformer
        x_train_reshaped = x_train_split.values.astype(np.float32).reshape(x_train_split.shape[0], 1, x_train_split.shape[1])
        x_val_reshaped = x_val.values.astype(np.float32).reshape(x_val.shape[0], 1, x_val.shape[1])
        y_train_split = y_train_split.values.astype(np.float32)
        y_val = y_val.values.astype(np.float32)

        # Create Transformer model (functional API for better stability with MultiHeadAttention)
        inputs = tf.keras.layers.Input(shape=(1, x_train.shape[1]))
        attention = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=32)(inputs, inputs)
        pooling = tf.keras.layers.GlobalAveragePooling1D()(attention)
        dense1 = tf.keras.layers.Dense(64, activation='relu')(pooling)
        outputs = tf.keras.layers.Dense(1)(dense1)

        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        model.compile(optimizer='adam', loss='mse')

        # Train
        epochs = self.config_loader.REDUCED_EPOCHS or 10
        history = model.fit(
            x_train_reshaped, y_train_split,
            epochs=epochs,
            validation_data=(x_val_reshaped, y_val),
            verbose=0
        )

        # Save model
        model_path = self.path_manager.batch_dir / f"model_{ticker}_{target_col}_transformer.keras"
        model.save(model_path)
        
        # Log to MLflow
        self._log_mlflow_run(
            ticker, target_col, "transformer",
            params={"head_size": 128, "num_heads": 4, "epochs": epochs},
            metrics={"loss": history.history['loss'][0]},
            artifact_path=str(model_path)
        )

        print(f"    🎯 Transformer - Loss: {history.history['loss'][-1]:.6f} - збережено: {model_path.name}")
        return {'loss': float(history.history['loss'][-1])}

    def _train_tabnet_model(self, x_train, y_train, ticker, target_col):
        """Тренування TabNet моделі"""
        try:
            import torch
            from pytorch_tabnet.tab_model import TabNetRegressor
            from sklearn.metrics import mean_squared_error
            from sklearn.model_selection import train_test_split

            # Split data
            x_train_split, x_val, y_train_split, y_val = train_test_split(
                x_train, y_train, test_size=0.2, random_state=42
            )

            # Create TabNet model
            model = TabNetRegressor(
                n_d=64, n_a=64,
                n_steps=3,
                gamma=1.5,
                lambda_sparse=1e-3,
                optimizer_fn=torch.optim.Adam,
                optimizer_params={"lr": 2e-2},
                mask_type='entmax',
                scheduler_params={"step_size":10, "gamma":0.9},
                verbose=0
            )

            # Train
            max_epochs = self.config_loader.REDUCED_EPOCHS or 20
            model.fit(
                X_train=x_train_split.values, 
                y_train=y_train_split.values.reshape(-1, 1),
                eval_set=[(x_val.values, y_val.values.reshape(-1, 1))],
                max_epochs=max_epochs,
                patience=5,
                batch_size=1024,
                virtual_batch_size=128,
                num_workers=0,
                drop_last=False
            )

            # Save model
            model_path = self.path_manager.batch_dir / f"model_{ticker}_{target_col}_tabnet.zip"
            model.save_model(str(model_path))

            # Evaluate
            y_pred = model.predict(x_val.values)
            mse = mean_squared_error(y_val, y_pred)

            print(f"    🎯 TabNet - MSE: {mse:.6f} - збережено: {model_path.name}")
            return {'mse': float(mse)}

        except ImportError:
            print("    ⚠️ TabNet не встановлено, пропускаємо")

    def _train_autoencoder_model(self, x_train, y_train, ticker, target_col):
        """Тренування Autoencoder моделі"""
        import tensorflow as tf
        from sklearn.model_selection import train_test_split

        # Split data
        # Split data
        x_train_split, x_val, y_train_split, y_val = train_test_split(
            x_train, y_train, test_size=0.2, random_state=42
        )

        # Ensure numpy float32
        x_train_np = x_train_split.values.astype(np.float32)
        x_val_np = x_val.values.astype(np.float32)
        y_train_np = y_train_split.values.astype(np.float32)
        y_val_np = y_val.values.astype(np.float32)

        # Create autoencoder model
        input_dim = x_train.shape[1]
        encoding_dim = 32

        # Encoder
        input_layer = tf.keras.layers.Input(shape=(input_dim,))
        encoder = tf.keras.layers.Dense(encoding_dim, activation='relu')(input_layer)

        # Regression head
        regression = tf.keras.layers.Dense(64, activation='relu')(encoder)
        output = tf.keras.layers.Dense(1)(regression)

        # Create model
        model = tf.keras.Model(inputs=input_layer, outputs=output)

        model.compile(optimizer='adam', loss='mse')

        # Train
        epochs = self.config_loader.REDUCED_EPOCHS or 50
        history = model.fit(
            x_train_np, y_train_np,
            epochs=epochs,
            batch_size=32,
            shuffle=True,
            validation_data=(x_val_np, y_val_np),
            verbose=0
        )

        # Save model
        model_path = self.path_manager.batch_dir / f"model_{ticker}_{target_col}_autoencoder.keras"
        model.save(model_path)

        print(f"    🎯 Autoencoder - Loss: {history.history['loss'][-1]:.6f} - збережено: {model_path.name}")
        return {'loss': float(history.history['loss'][-1])}

    def _get_model_max_features(self, model_type):
        """Отримати максимальну кількість фіч для моделі"""
        max_features_map = {
            'mlp': 256,
            'lstm': 128,
            'gru': 128,
            'cnn': 64,
            'transformer': 128,
            'tabnet': 256,
            'autoencoder': 128,
            'random_forest': 256
        }
        return max_features_map.get(model_type.lower(), 128)

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    print("🔥 COLAB TRAINING CONTROLLER - CLEAN VERSION")
    print("="*80)

    # Initialize controller first to setup paths
    controller = ColabTrainingController()

    # ВИПРАВИТИ DTYPE ПРОБЛЕМУ
    fix_neural_models_dtype()

    # ОЧИСТИТИ КЕШ SmartFeatureSelector
    try:
        from src.features.selection.smart_selector import SmartFeatureSelector
        if hasattr(SmartFeatureSelector, '_feature_cache'):
            setattr(SmartFeatureSelector, '_feature_cache', {})
        if hasattr(SmartFeatureSelector, '_model_cache'):
            setattr(SmartFeatureSelector, '_model_cache', {})
        print("🧹 Кеш SmartFeatureSelector очищено")
    except Exception:
        logger.warning("Failed to clear cache or session", exc_info=True)

    # ОЧИСТИТИ ФАЙЛОВИЙ КЕШ В COLAB
    try:
        import shutil
        colab_cache_path = Path("/content/drive/MyDrive/trading_project/cache/unified_cache")
        if colab_cache_path.exists():
            shutil.rmtree(colab_cache_path)
            print("🧹 Файловий кеш в Colab очищено")
        else:
            print("📁 Файловий кеш в Colab не знайдено")
    except Exception as e:
        print(f"⚠️ Помилка очищення файлового кешу: {e}")

    # Run initialization
    batch_dir = controller.initialize()

    # Run training pipeline
    controller.run_training_pipeline()

    print("\n✅ COLAB TRAINING COMPLETED!")