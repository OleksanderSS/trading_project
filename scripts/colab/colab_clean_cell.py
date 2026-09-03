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

from src.config.target_type_registry import (  # noqa: E402
    CLASSIFICATION_BINARY_TYPE,
    CLASSIFICATION_MULTICLASS_TYPE,
    CLASSIFICATION_TARGET_TYPES,
    load_target_types,
)

warnings.filterwarnings('ignore')

# Target type taxonomy re-exported from src/config/target_type_registry.py
# (the single source of truth, shared with the live pipeline's champion
# selector) so `from scripts.colab.colab_clean_cell import
# CLASSIFICATION_BINARY_TYPE` etc. keeps working for existing callers/tests.

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
# NOTE: the DTYPE FIX block that used to sit here has been removed.
#
# It monkeypatched .train on BaseNeuralModel/CNNModel/LSTMModel/GRUModel and
# .fit on TransformerModel. This cell instantiates none of them -- those five
# names appeared nowhere else in this file. Every model here is built
# directly: sklearn MLPRegressor/MLPClassifier, tf.keras.Sequential for
# cnn/lstm/gru/transformer/autoencoder, pytorch_tabnet for tabnet. So the
# patches applied to classes nothing in this process called, and it printed
# "ВСІ НЕЙРОННІ МОДЕЛІ ВИПРАВЛЕНО" for work that had no effect.
#
# Two reasons to remove it rather than leave it sitting harmless:
#
# 1. The BaseNeuralModel patch was broken. It called the original as
#    original_train(self, x, y, epochs, batch_size, validation_split), but
#    the signature is train(self, X, y, **kwargs) -- reproduced locally:
#    "BaseNeuralModel.train() takes 3 positional arguments but 6 were given".
#    MLPModel does not override train, so anything reaching for the project's
#    MLP in this process would have died on it. A patch that breaks the thing
#    it claims to fix is worse than absent.
# 2. The dtype conversion it added is already in the classes themselves
#    (base_neural.py: X.values / np.asarray then .astype(np.float32)), so
#    even the working half was redundant.
#
# What is NOT carried over, deliberately: np.nan_to_num(y, nan=0.0). Turning
# a missing TARGET into 0.0 does not clean data, it invents a label -- "no
# return" is a perfectly plausible training example, so the model learns from
# a value nobody measured. Rows with a missing target must be dropped, not
# filled.
# ==============================================================================
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
        self.target_types: dict[str, str] = self._load_target_types()
        self.heavy_models: list[str] = self._load_heavy_models()

    #: Last resort only. Kept in sync with models.yaml by
    #: tests/contracts/test_hybrid_split_single_source.py, which fails if the
    #: two drift -- a fallback that silently disagrees with the config is
    #: worse than no fallback, because it looks like it worked.
    _HEAVY_MODELS_FALLBACK = [
        'mlp', 'cnn', 'lstm', 'gru', 'transformer', 'tabnet', 'autoencoder',
    ]

    def _load_heavy_models(self) -> list[str]:
        """Read the heavy-model list from src/config/models.yaml.

        This list used to be a literal inside run_training_pipeline, which
        made it a second hand-maintained copy of models.yaml's
        `models.categories.heavy`. They agreed, but nothing kept them
        agreeing -- and the local side had already been annotated to say
        models.yaml is the source. One copy is now the source and this reads
        it, the same way _load_target_types reads targets.yaml.

        Falls back rather than raising: this runs in Colab with booked GPU
        time, and an incompletely synced repo should not cost a session. The
        warning is loud because a silent fallback is how the two drifted
        apart in the first place.
        """
        models_path = self.project_path / "src" / "config" / "models.yaml"
        if not models_path.exists():
            print(f"⚠️ {models_path} не знайдено — використовую вбудований список важких моделей")
            return list(self._HEAVY_MODELS_FALLBACK)
        try:
            import yaml
            with open(models_path, encoding="utf-8") as handle:
                config = yaml.safe_load(handle) or {}
            heavy = (
                (config.get("models") or {}).get("categories", {}) or {}
            ).get("heavy")
            if not isinstance(heavy, list) or not heavy:
                print(f"⚠️ У {models_path} немає models.categories.heavy — використовую вбудований список")
                return list(self._HEAVY_MODELS_FALLBACK)
            return [str(name) for name in heavy]
        except (OSError, yaml.YAMLError, AttributeError, TypeError) as exc:
            print(f"⚠️ Не вдалося прочитати {models_path} ({exc}) — використовую вбудований список")
            return list(self._HEAVY_MODELS_FALLBACK)

    def _load_target_types(self) -> dict[str, str]:
        """Read each target's declared `type:` from src/config/targets.yaml
        via the shared registry loader (src/config/target_type_registry.py)
        -- the live pipeline's champion selector reads the same file the
        same way, so the two can never disagree on what a target is.

        Every model-training function needs to know whether a target is
        classification_binary/classification_multiclass (needs a sigmoid/
        softmax head, cross-entropy loss, accuracy/AUC -- never MSE) or
        regression/indicator_prediction (needs a scaled continuous target
        and MSE) BEFORE this fix, every target was silently trained as
        plain regression regardless of what it actually was.
        """
        targets_path = self.project_path / "src" / "config" / "targets.yaml"
        if not targets_path.exists():
            print(f"⚠️ {targets_path} не знайдено -- усі таргети тренуватимуться як regression")
            return {}
        types = load_target_types(targets_path)
        if not types:
            print(f"⚠️ Не вдалося прочитати {targets_path} -- усі таргети тренуватимуться як regression")
        return types

    def target_type_for(self, target_col: str) -> str:
        return self.target_types.get(target_col, "regression")

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

        # Set batch directory.
        #
        # COLAB_BATCH_DIR wins when set, because the batch does not always
        # live inside the checkout. One real workflow copies features.parquet
        # and targets.parquet from a local machine to Drive, trains against
        # them there, and copies the results back -- so the code is in one
        # place and the data in another, and deriving the batch path from
        # PROJECT_PATH finds nothing.
        #
        # Falls back to the in-project location, which is where --mode
        # prepare writes it.
        override = os.environ.get("COLAB_BATCH_DIR")
        self.batch_dir = self._resolve_batch_dir(override)

    @staticmethod
    def _holds_batch(path) -> bool:
        path = Path(path)
        return (path / "features.parquet").exists() and (path / "targets.parquet").exists()

    def _resolve_batch_dir(self, override: str | None) -> Path:
        """Where the batch actually is: checked, then searched, then guessed.

        An override used to be taken on faith. A run then failed several
        screens later with

            FileNotFoundError: Файли відсутні в
            /content/drive/MyDrive/.../main_database

        -- literal ellipsis, because the placeholder in a copied instruction
        looked like a path. Trusting an override that points at nothing and
        discovering it in the data loader is the worst of both: the value is
        wrong AND the message arrives far from where it was set.

        An override that holds no batch is now REJECTED at the point it is
        read, and the search continues rather than the run failing outright.
        """
        default = Path(self.PROJECT_PATH) / "data" / "colab" / "accumulated" / "main_database"

        if override:
            candidate = Path(override)
            if self._holds_batch(candidate):
                print(f"📦 BATCH_DIR (from COLAB_BATCH_DIR): {candidate}")
                return candidate
            print(
                f"⚠️ COLAB_BATCH_DIR points at {candidate}, which holds no "
                "features.parquet/targets.parquet — ignoring it and looking "
                "elsewhere."
            )

        if self._holds_batch(default):
            print(f"📦 BATCH_DIR: {default}")
            return default

        # Any other batch under the project's accumulated directory, newest
        # first: a batch is identifiable by what it contains, so there is no
        # need to make anyone type a path.
        root = Path(self.PROJECT_PATH) / "data" / "colab" / "accumulated"
        if root.exists():
            found = sorted(
                (p for p in root.iterdir() if p.is_dir() and self._holds_batch(p)),
                key=lambda p: (p / "features.parquet").stat().st_mtime,
                reverse=True,
            )
            if found:
                print(f"📦 BATCH_DIR (found): {found[0]}")
                if len(found) > 1:
                    print(f"   ({len(found)} batches present; took the newest)")
                return found[0]

        print(f"❌ No batch found. Looked in {default}" +
              (f", and under {root}" if root.exists() else "") +
              (f", and at COLAB_BATCH_DIR={override}" if override else "") + ".")
        print("   A batch is a directory holding features.parquet and targets.parquet.")
        return default

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

            # From src/config/models.yaml (models.categories.heavy), not a
            # literal: this was a second hand-maintained copy of the config
            # the local side already treats as the source of the split.
            heavy_models = self.config_loader.heavy_models
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

        # Merge data.
        #
        # The key is (ticker, datetime, interval). It used to omit interval,
        # and the export stacks all three timeframes into one file -- so a
        # 15m bar at 15:30 and a 60m bar at 15:30 are two rows sharing a
        # key. Measured on the 2026-08-06 batch: 2,550 of 26,989 rows
        # duplicate on (ticker, datetime), and zero duplicate once interval
        # joins the key. validate='one_to_one' caught it, which is what it
        # is there for; the crash was the diagnosis, not the disease.
        common_cols = ['ticker']
        for key in ('datetime', 'interval'):
            if key in t_feat.columns and key in t_targ.columns:
                common_cols.append(key)

        merged = pd.merge(t_feat, t_targ, on=common_cols, how='inner', validate='one_to_one')
        print(f"  ✅ Merged: {merged.shape} on {common_cols}")

        target_cols = [c for c in merged.columns if c.startswith('target_')]
        if self.config_loader.TEST_TARGET:
            if self.config_loader.TEST_TARGET in target_cols:
                target_cols = [self.config_loader.TEST_TARGET]

        # One model per timeframe, not one model over all of them.
        #
        # Measured on the 2026-08-06 export, the targets are almost perfectly
        # partitioned by timeframe: target_return_1d exists only on 1d rows,
        # target_intraday_return_15m only on 15m. So the old pooled fit was
        # mostly self-selecting -- mask = notna() left one timeframe's rows
        # standing anyway. Two things were still wrong, and neither was
        # visible:
        #
        # - target_hourly_return_1h, _up_1h and _volume_spike_1h are
        #   populated on BOTH 15m and 60m rows (10,234 and 3,054). Those
        #   three genuinely mixed two bar sizes into one fit, with `interval`
        #   dropped as a string column so the model could not tell them
        #   apart.
        # - Every model was recorded under the timeframe 'all'. Stage 5
        #   selects prediction rows by the model's timeframe, so a label that
        #   names no timeframe cannot be matched to the rows it should score.
        #
        # The local light branch has always keyed champions by
        # {ticker}_{timeframe}_{target}_{pattern}. This makes the heavy
        # branch agree with it.
        for timeframe, tf_rows in self._by_timeframe(merged):
            if 'datetime' in tf_rows.columns:
                # Sequence models need genuine chronological order to build
                # real historical windows -- merge's output row order isn't a
                # documented guarantee, so this must be explicit, not assumed.
                tf_rows = tf_rows.sort_values('datetime')
            tf_rows = tf_rows.reset_index(drop=True)

            # Only the targets this timeframe actually carries.
            #
            # The list was taken once from the whole merged frame, so each
            # timeframe then walked all 27 and announced "0 samples" for the
            # ~20 that do not exist at that cadence -- target_up_1d has no
            # meaning on a 15-minute bar, and the targets ARE partitioned
            # that way by design, not by accident. Cheap in time (the sample
            # check returns before any work) but it buried the real lines in
            # the log and wrote an empty entry per skipped target into
            # colab_results.json.
            live_targets = [c for c in target_cols
                            if tf_rows[c].notna().sum() >= self._MIN_TRAINING_SAMPLES]
            skipped = len(target_cols) - len(live_targets)
            print(f"\n  ⏱️  Таймфрейм {timeframe}: {tf_rows.shape} | "
                  f"таргетів {len(live_targets)}"
                  + (f" (ще {skipped} не існують на цьому таймфреймі)" if skipped else ""))

            for target_col in live_targets:
                self._process_target(ticker, timeframe, target_col, tf_rows, heavy_models)

    @staticmethod
    def _by_timeframe(merged):
        """(timeframe, rows) pairs -- one pair when the column is absent.

        An export without `interval` still trains, under the label 'all',
        rather than being silently skipped. That is the honest name for a
        frame whose timeframe is unknown, and it is what the results file
        used to record for every model regardless.
        """
        if 'interval' not in merged.columns:
            return [('all', merged)]
        return [(str(tf), rows) for tf, rows in merged.groupby('interval', sort=True)]

    @classmethod
    def _classification_split_verdict(cls, y_ser):
        """Why this classification target cannot be trained here, or None.

        The split is chronological, so a rare event that happens to cluster
        late lands entirely in validation and leaves the training portion
        with a single class. Measured on AAPL 60m
        target_hourly_breakout_1h: 11 positives in 278 rows, all 11 inside
        the final 20%.

        Six of the seven trainers accepted that without complaint. A
        classifier fitted to one class can only ever predict that class,
        and it scored 44/55 = 80% on a validation window that is mostly the
        same class -- a respectable-looking number for a model that learned
        nothing. Only TabNet refused, with

            Valid set -- {0, 1} -- contains unkown targets from training set

        so the one model that failed loudly was the only honest report in
        the group, and the six that "succeeded" were the real damage.

        Checked once for the whole target rather than per trainer: the
        condition is a property of the data and the split, not of any
        architecture. Sequence models purge additional rows off the end of
        the training portion, so they can still hit a degenerate split this
        does not catch -- that now surfaces as a recorded error rather than
        a success, which is the other half of the same fix.

        Rare-but-present is left to train. Where the line sits between
        "rare" and "too rare" is a modelling decision, not something to
        decide silently inside a guard; the imbalance is printed so it is
        visible in the log.
        """
        y = pd.Series(y_ser).dropna()
        if len(y) < cls._MIN_TRAINING_SAMPLES:
            return f"лише {len(y)} зразків"

        _, _, y_train, y_val = cls._chronological_split(y, y)
        train_classes = sorted(pd.Series(y_train).unique().tolist())
        if len(train_classes) < 2:
            counts = pd.Series(y).value_counts().sort_index().to_dict()
            return (
                f"тренувальна вибірка має один клас {train_classes} — "
                f"модель могла б передбачати лише його "
                f"(усього {counts}, розділ хронологічний)"
            )
        return None

    def _results_slot(self, ticker, timeframe, target_col):
        """The dict this (ticker, timeframe, target) writes its models into."""
        return (
            self.results['ticker_results']
            .setdefault(ticker, {'timeframes': {}})['timeframes']
            .setdefault(timeframe, {'results': {}})['results']
            .setdefault(target_col, {'models': {}})['models']
        )

    def _process_target(self, ticker, timeframe, target_col, merged, heavy_models):
        """Обробка одного цільового стовпця"""
        print(f"\n  🎯 Таргет: {target_col} [{timeframe}]")

        self._results_slot(ticker, timeframe, target_col)

        # Filter data
        mask = merged[target_col].notna()
        if mask.sum() < self._MIN_TRAINING_SAMPLES:
            print(f"    ⚠️ Лише {mask.sum()} зразків, занадто мало.")
            return

        print(f"    📊 Data size: {mask.sum()} samples, {len(merged.columns)} columns")

        # Prepare training data. `interval` goes with the identity columns:
        # within this call it is a constant, so it carries no signal, and
        # leaving it to be dropped later by dtype is an accident waiting to
        # be undone by an encoder.
        x_df = merged.loc[mask].drop(
            columns=['ticker', 'datetime', 'interval']
            + [c for c in merged.columns if c.startswith('target_')],
            errors='ignore',
        )
        y_ser = merged.loc[mask, target_col]

        # Process data types
        # Process data types - CRITICAL FIX FOR DTYPE OBJECT
        x_df = x_df.select_dtypes(exclude=['datetime64', 'datetime', 'object'])
        x_df = x_df.apply(pd.to_numeric, errors='coerce').fillna(0).astype(np.float32)
        x_df = x_df.replace([np.inf, -np.inf], 0)

        y_ser = pd.to_numeric(y_ser, errors='coerce').fillna(0).astype(np.float32)

        # Target type: classification_binary/classification_multiclass targets
        # are already integer-coded labels (0/1 or 0/1/2) and must never be
        # scaled or trained with MSE; regression/indicator_prediction targets
        # are continuous and get their own StandardScaler (mirroring X) so a
        # price-level target like SMA/EMA doesn't dominate the loss purely
        # from its raw unit magnitude. src/config/targets.yaml is the source
        # of truth -- this script used to ignore it entirely and train every
        # target, including binary up/down targets, as plain regression.
        target_type = self.config_loader.target_type_for(target_col)
        is_classification = target_type in CLASSIFICATION_TARGET_TYPES
        y_scaler = None

        if is_classification:
            verdict = self._classification_split_verdict(y_ser)
            if verdict:
                print(f"    ⛔ {verdict}")
                self._results_slot(ticker, timeframe, target_col)['_context'] = {
                    'status': 'error', 'message': verdict,
                }
                return

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

        if not is_classification:
            try:
                from sklearn.preprocessing import StandardScaler
                y_scaler = StandardScaler()
                y_scaled_values = y_scaler.fit_transform(y_ser.to_numpy().reshape(-1, 1)).ravel()
                y_ser = pd.Series(y_scaled_values.astype(np.float32), index=y_ser.index, name=y_ser.name)
                print("    ⚖️ Таргет масштабовано (StandardScaler)")
            except Exception as e:
                print(f"    ⚠️ Помилка масштабування таргету: {e}")
                y_scaler = None

        # Train models
        for model_type in heavy_models:
            self._train_model(ticker, timeframe, target_col, model_type, x_df, y_ser, target_type, is_classification, y_scaler)

    @staticmethod
    def _context_id(ticker, timeframe, target_col, model_type):
        """The name this model is known by, everywhere.

        The timeframe used to be absent from both the filename and the
        metadata key, which was survivable only while one model covered all
        three timeframes. Split by timeframe and the old key names three
        different models identically -- and model_resolver assigns into
        models_metadata[key], so two of the three would be overwritten and
        lost without a word.

        Field order matches the light branch's champion key,
        {ticker}_{timeframe}_{target}_{...}, so both halves of the hybrid
        read the same way.

        Delegates to src.pipeline.constants.heavy_model_key rather than
        repeating the format string. This WAS a second copy: the constant's
        own docstring says "the Colab writer builds the same string in
        _context_id", which records the duplication without preventing it.
        They agreed, but a key that two files build independently is one edit
        away from silently producing models Stage 5's '*{context_id}*' glob
        cannot find — and a model that cannot be found fails no test.

        Falls back to the literal if the repo is not importable, which is the
        Colab-with-booked-GPU case that _load_heavy_models also guards.
        """
        try:
            from src.pipeline.constants import heavy_model_key
            return heavy_model_key(ticker, timeframe, target_col, model_type)
        except ImportError:
            return f"{ticker}_{timeframe}_{target_col}_{model_type}"

    def _train_model(self, ticker, timeframe, target_col, model_type, x_df, y_ser, target_type, is_classification, y_scaler):
        """Тренування однієї моделі"""
        # Check if model already exists to skip re-training
        ext = ".keras" if model_type in ['cnn', 'lstm', 'gru', 'transformer', 'autoencoder'] else ".pkl"
        if model_type == 'tabnet': ext = ".zip"

        context_id = self._context_id(ticker, timeframe, target_col, model_type)
        model_filename = f"model_{context_id}{ext}"
        model_path = self.path_manager.batch_dir / model_filename
        
        if model_path.exists():
            # A skipped model used to be recorded with
            # metrics={'info': 'already_exists'} and status='success'. The
            # numbers from the run that actually trained it were gone, so on
            # every re-run the whole heavy branch reported no measured
            # quality at all: 1,405 of 1,638 contexts in the 2026-05-10
            # results carry that placeholder and nothing else.
            #
            # That is not a cosmetic loss. Stage 5 ranks candidates on r2 or
            # accuracy and scores a model with no usable metric at -inf, so a
            # skipped heavy model can never become a champion -- the skip
            # silently disqualifies it.
            sidecar = self._load_sidecar(model_filename)

            # Reuse only when the model was fitted to THIS batch.
            #
            # "The file exists" was the entire cache test, and a filename
            # carries no trace of the data behind it. Re-run prepare -- new
            # bars, a changed indicator, a different ticker set -- and every
            # model keeps its name while the features underneath it move.
            # The old check would skip all of them forever, so the batch
            # would silently be scored by models fitted to data it no longer
            # contains.
            current_batch = self._batch_fingerprint()
            trained_on = (sidecar or {}).get("batch_fingerprint", "")
            if sidecar and current_batch and trained_on and trained_on != current_batch:
                print(
                    f"    🔄 {model_type:<14} | Модель від іншої партії "
                    f"({trained_on[:8]}… ≠ {current_batch[:8]}…) — перетреновую."
                )
                sidecar = None

            if sidecar:
                print(f"    ⏭️  {model_type:<14} | Вже існує, метрики з кешу.")
                model_result = {
                    'status': 'success',
                    'model_path': model_filename,
                    'metrics': sidecar.get('metrics', {}),
                    'selected_features': sidecar.get('selected_features', []),
                }
                self._results_slot(ticker, timeframe, target_col)[model_type] = model_result
                self.results.setdefault('models_metadata', {})[context_id] = {
                    'ticker': ticker, 'target': target_col,
                    'timeframe': timeframe,
                    'model_type': model_type, 'model_path': model_filename,
                    'metrics': sidecar.get('metrics', {}),
                    'selected_features': sidecar.get('selected_features', []),
                }
                return model_result

            print(f"    ⏭️  {model_type:<14} | Вже існує, метрик немає — пропускаю.")

            # Record result for skipped model
            try:
                max_features = self._get_model_max_features(model_type)
                selected_features = self.feature_selector.select_features(
                    X=x_df, y=y_ser, context_id=context_id,
                    is_classification=is_classification, max_features=max_features
                )
            except Exception as e:
                # An unknown feature list is not "every column".
                #
                # This branch only re-derives the selection for a model that
                # already exists on disk, so a failure here means we do not
                # KNOW what the model was trained on. Writing every column
                # into `selected_features` hands Stage 5 a list of 1,384
                # names for a model fitted on five, and Stage 5 believes it:
                # it prepares that many columns and either wastes the work or
                # refuses the context for missing features. Recording the
                # failure is the honest answer, and Stage 5 already knows how
                # to skip a context it cannot serve --
                # `missing selected features; skipping prediction instead of
                # filling zeros`.
                self.logger.error(
                    "Feature selection failed for an existing model; "
                    "recording it as an error rather than claiming every "
                    "column was used.", exc_info=True,
                )
                self._results_slot(ticker, timeframe, target_col)[model_type] = {
                    'status': 'error',
                    'message': f'feature selection failed: {str(e)[:160]}',
                }
                return

            model_result = {
                'status': 'success',
                'model_path': model_filename,
                'metrics': {'info': 'already_exists'},
                'selected_features': selected_features
            }
            
            self._results_slot(ticker, timeframe, target_col)[model_type] = model_result

            # Add to models_metadata. Key is 'model_path', not 'path' --
            # every downstream consumer (model_resolver.py, prediction/
            # orchestrator.py, scaler_service.py, data_preparer.py,
            # result_builder.py) reads meta.get('model_path'); a 'path' key
            # here is invisible to all of them, silently forcing Stage 5
            # onto its slower filename-glob fallback instead of the direct
            # path, and skipping ResultsProcessor._convert_model_paths'
            # localization step entirely.
            self.results['models_metadata'][context_id] = {
                'ticker': ticker,
                'target': target_col,
                'timeframe': timeframe,
                'model_type': model_type,
                'model_path': model_filename,
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
                context_id=context_id,
                is_classification=is_classification,
                max_features=max_features
            )

            if len(selected_features) == 0:
                print("❌ 0 фіч вибрано")
                return

            print(f"✅ OK ({len(selected_features)} фіч)")

            # Train the model
            metrics = self._train_model_with_features(
                ticker, target_col, model_type, x_df, y_ser, selected_features,
                target_type, is_classification, y_scaler, model_path,
            )

            # model_filename was recomputed here, by a second copy of the
            # extension table and the naming expression. Two copies of a name
            # is one copy too many -- the value from the top of this method
            # is the one the skip check and _save_sidecar already use.

            # A trainer that RETURNS an error is still a failure.
            #
            # Several trainers report failure by returning {'error': ...}
            # rather than raising: TabNet when pytorch_tabnet is missing, the
            # sequence models when there is not enough history to build a
            # single window. Nothing here looked at that, so the run recorded
            # status='success', wrote a metrics sidecar, and listed a
            # model_path for a file that was never written.
            #
            # Measured on the 2026-08-07 run: 4,620 sidecars against 4,619
            # models -- the extra one is exactly this. The count is how the
            # defect was found, which is the point of counting.
            if isinstance(metrics, dict) and 'error' in metrics:
                print(f"    ❌ {model_type:<14} | {metrics['error']}")
                self._results_slot(ticker, timeframe, target_col)[model_type] = {
                    'status': 'error',
                    'message': str(metrics['error'])[:200],
                }
                return

            if not model_path.exists():
                # The trainer reported numbers and left no file. Recording it
                # as a success would hand Stage 5 a path to nothing.
                print(f"    ❌ {model_type:<14} | метрики є, файл моделі не записано")
                self._results_slot(ticker, timeframe, target_col)[model_type] = {
                    'status': 'error',
                    'message': f'no model file at {model_filename}',
                }
                return

            model_result = {
                'status': 'success',
                'model_path': model_filename,
                'metrics': metrics,
                'selected_features': selected_features
            }

            # Persist the metrics beside the model. Without this the numbers
            # exist only in THIS run's results file, and the next run --
            # which skips the model because its file is on disk -- has
            # nothing to report but "already_exists". See _load_sidecar.
            self._save_sidecar(model_filename, metrics, selected_features)

            self._results_slot(ticker, timeframe, target_col)[model_type] = model_result

            # Add to models_metadata for easier access. 'model_path', not
            # 'path' -- see the matching comment in the skipped-model branch
            # above for why this key name matters.
            self.results['models_metadata'][context_id] = {
                'ticker': ticker,
                'target': target_col,
                'timeframe': timeframe,
                'model_type': model_type,
                'model_path': model_filename,
                'metrics': metrics,
                'selected_features': selected_features
            }

        except Exception as e:
            print(f"❌ Помилка: {str(e)[:100]}")
            # Record error. This used to be guarded by two membership tests
            # that silently dropped the record when either was false -- so a
            # failure during the very first target of a ticker went unlogged,
            # which is exactly the failure worth having in the report.
            self._results_slot(ticker, timeframe, target_col)[model_type] = {
                'status': 'error',
                'message': str(e)[:100]
            }

    def _train_model_with_features(
        self, ticker, target_col, model_type, x_df, y_ser, selected_features,
        target_type, is_classification, y_scaler, model_path,
    ):
        """Тренування моделі з вибраними фічами.

        `model_path` is passed in rather than rebuilt. Each of the seven
        trainers used to compose its own filename from ticker and target,
        which was a second (and third, and eighth) definition of a name the
        caller had already decided. The skip check, _save_sidecar and the
        results file all use the caller's version; a trainer that composed
        it even slightly differently would write a model nobody could find
        again, and be retrained on every run forever.

        It stopped being hypothetical the moment the timeframe joined the
        name: nine call sites would have had to change together.
        """
        try:
            # Prepare data with selected features
            x_train = x_df[selected_features]
            # Computed once here (while x_df still has real column names,
            # before any model-specific selection/windowing) so every
            # trainer reports the same context regardless of its own
            # feature subset or data shape -- see _context_windows.
            context_windows = self._context_windows(x_df)
            kwargs = dict(is_classification=is_classification, y_scaler=y_scaler,
                          context_windows=context_windows, model_path=model_path)

            if model_type in self._SEQUENCE_MODEL_TYPES:
                x_seq, y_seq = self._build_sequences(x_train, y_ser, self._SEQUENCE_WINDOW)
                if x_seq is None:
                    msg = f'insufficient history for sequence window {self._SEQUENCE_WINDOW} ({len(x_train)} rows)'
                    print(f"⚠️ {msg}")
                    return {'error': msg}
                if model_type == 'cnn':
                    return self._train_cnn_model(x_seq, y_seq, ticker, target_col, **kwargs)
                elif model_type == 'lstm':
                    return self._train_lstm_model(x_seq, y_seq, ticker, target_col, **kwargs)
                elif model_type == 'gru':
                    return self._train_gru_model(x_seq, y_seq, ticker, target_col, **kwargs)
                elif model_type == 'transformer':
                    return self._train_transformer_model(x_seq, y_seq, ticker, target_col, **kwargs)

            # Create model based on type. Every branch here used to call its
            # trainer without `return`, so a *successful* run's real metrics
            # dict was silently discarded and replaced with None -- only the
            # except-block's {'error': ...} ever actually reached the caller.
            if model_type == 'mlp':
                return self._train_mlp_model(x_train, y_ser, ticker, target_col, **kwargs)
            elif model_type == 'tabnet':
                return self._train_tabnet_model(x_train, y_ser, ticker, target_col, **kwargs)
            elif model_type == 'autoencoder':
                return self._train_autoencoder_model(x_train, y_ser, ticker, target_col, **kwargs)
            else:
                print(f"⚠️ Невідомий тип моделі: {model_type}")
                return {'error': f'unknown model_type {model_type}'}

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
                timeframe = meta.get('timeframe', '')

                # Format required by HybridOrchestrator
                fs_data = {
                    'ticker': ticker,
                    'timeframe': timeframe,
                    'targets': [target],
                    'model_name': model,
                    'selected_features': meta['selected_features'],
                    'timestamp': datetime.now().isoformat()
                }
                
                # Named for the model it belongs to, which now includes the
                # timeframe. Built from ticker/target/model alone, the three
                # timeframes' feature sets were three writes to one path --
                # the same collision as the model files themselves, and just
                # as silent, since the readers match on the file's CONTENT
                # and would happily accept whichever copy survived.
                fs_filename = f"selected_features_{meta_key}.json"
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

    # ── target-type-aware helpers (shared by every trainer below) ──────────
    #
    # Before this fix, every trainer built a bare Dense(1) (linear) output,
    # compiled with loss='mse', and reported history.history['loss'] (the
    # TRAINING-set loss, not validation, despite validation_data being
    # passed to .fit()) or an sklearn MSE computed on scaled-but-unlabeled
    # y -- regardless of whether the target was a 0/1 label
    # (classification_binary), a 0/1/2 label (classification_multiclass),
    # or a genuinely continuous value (regression/indicator_prediction).
    # That's why there was no accuracy/AUC anywhere for target_up_1d etc.,
    # and why price-level targets (SMA/EMA/BB) showed MSE in the tens of
    # thousands -- an unscaled dollar-denominated target trained with a
    # loss function that has no notion of the target's own units.

    # cnn/lstm/gru/transformer get a real (window, n_features) history per
    # sample instead of a single flattened snapshot reshaped to a fake
    # sequence length of 1 -- a recurrent layer or attention block cannot
    # learn anything across a single timestep, so before this fix these
    # three architectures were paying real compute for no more expressive
    # power than a plain dense layer applied once. mlp/tabnet/autoencoder
    # are non-sequential by design and correctly stay on the flat features.
    #: Below this, a target is not trainable and is not attempted. One
    #: definition: the per-timeframe target list and the per-target guard
    #: both read it, and two copies of a threshold are two thresholds as
    #: soon as one of them is tuned.
    _MIN_TRAINING_SAMPLES = 50

    _SEQUENCE_WINDOW = 20  # trading days of history; matches the SMA_20/BB_20 lookback already used elsewhere in this pipeline
    _SEQUENCE_MODEL_TYPES = {'cnn', 'lstm', 'gru', 'transformer'}

    @staticmethod
    def _build_sequences(x_selected: pd.DataFrame, y_ser: pd.Series, window: int):
        """Turn flat, chronologically-ordered, single-ticker rows into
        overlapping (window, n_features) sequences. The target for each
        sequence is the target value on the sequence's LAST (most recent)
        day -- the target's own meaning (e.g. "up in the next 5 days") is
        unchanged; this only gives the model real prior days as context
        instead of one flattened snapshot. Returns (None, None) if there
        isn't enough history for even one full window.
        """
        values = x_selected.to_numpy(dtype=np.float32)
        if len(values) < window:
            return None, None
        windows = np.lib.stride_tricks.sliding_window_view(values, window, axis=0)
        # sliding_window_view appends the window axis last -> (n, features,
        # window); Keras wants (n, window, features).
        x_seq = np.ascontiguousarray(np.transpose(windows, (0, 2, 1)))
        y_seq = y_ser.to_numpy(dtype=np.float32)[window - 1:]
        return x_seq, y_seq

    @staticmethod
    def _chronological_split(x, y, val_fraction: float = 0.2, purge: int = 0):
        """Time-ordered train/validation split -- replaces the random
        train_test_split(..., random_state=42) used everywhere in this file.

        A random split lets validation rows sit chronologically before, or
        interleaved with, training rows -- already a lookahead concern for
        autocorrelated daily data, and far worse once cnn/lstm/gru/
        transformer started receiving overlapping 20-day windows: a random
        split could put a validation window sharing up to 19 of its 20 days
        with an adjacent training window, making validation metrics -- and
        therefore champion selection -- look better than genuine
        out-of-sample performance would. `purge` drops that many rows
        between train and validation to remove window-overlap leakage at
        the boundary entirely: pass `window - 1` for the sequence models,
        0 for flat single-row samples (mlp/tabnet/autoencoder), which have
        no such overlap to purge.

        Assumes `x`/`y` are already in chronological order (the caller is
        responsible for that -- see the explicit sort in _process_ticker
        and the ordering _build_sequences preserves).
        """
        n = len(x)
        val_size = max(1, int(n * val_fraction))
        val_start = n - val_size
        train_end = max(0, val_start - purge)
        if isinstance(x, pd.DataFrame):
            return x.iloc[:train_end], x.iloc[val_start:], y.iloc[:train_end], y.iloc[val_start:]
        return x[:train_end], x[val_start:], y[:train_end], y[val_start:]

    # ── windowed validation reporting (metric + market context per slice) ──
    #
    # One aggregate metric from one validation split hides two things: (a)
    # whether a model was consistently good or just got lucky/unlucky in
    # one stretch of the period, and (b) whether performance correlates
    # with market conditions at all -- the question behind the "context
    # map" idea. Slicing the *already-computed* validation predictions into
    # a few contiguous chronological windows, and reading a context
    # snapshot for each window, answers both without any extra training:
    # this is purely a richer way to report on the one train/val split
    # every trainer already does. Full regime-conditioned model *selection*
    # (a champion per regime, not just per ticker/target) is deliberately
    # NOT implemented here -- a single validation window per model doesn't
    # give enough regime-labeled observations to trust that comparison yet,
    # and ModelSelectionService.select_best_model_for_context() already
    # owns regime-based selection at prediction time; this only attaches
    # evidence a future version of that could eventually be fed with.

    _VALIDATION_REPORT_WINDOWS = 3
    # Reuses columns MarketContextAnalyzer already computes
    # (src/analytics/context/market_context_analyzer.py) -- matched by
    # prefix since the real column names carry a timeframe suffix (e.g.
    # "market_context_volatility_ratio_1d"). Never recomputes context from
    # raw prices here; there is exactly one place in this codebase that
    # defines what "market context" means.
    _CONTEXT_COLUMN_PREFIXES = (
        "market_context_volatility_ratio",
        "market_context_trend_20d",
        "market_context_rsi_current",
        "market_context_volume_ratio",
        "market_context_market_breadth",
        "market_context_yield_curve_slope",
        "market_context_yield_curve_inverted",
    )

    @classmethod
    def _context_snapshot(cls, x_window: pd.DataFrame) -> dict:
        """Mean of each available market_context_* column over one window."""
        snapshot = {}
        for prefix in cls._CONTEXT_COLUMN_PREFIXES:
            matching = [c for c in x_window.columns if c.startswith(prefix)]
            if matching:
                value = pd.to_numeric(x_window[matching[0]], errors="coerce").mean()
                if pd.notna(value):
                    snapshot[prefix] = float(value)
        return snapshot

    @classmethod
    def _context_windows(
        cls, x_df: pd.DataFrame, val_fraction: float = 0.2, n_windows: int | None = None,
    ) -> list[dict]:
        """Context snapshot for each of n_windows contiguous chronological
        slices of the validation portion (last val_fraction of x_df).

        Computed once from the FULL feature set (not a model's
        selected_features), so it's identical across every model_type for
        the same (ticker, target) regardless of which columns that
        particular model happened to select. Purely descriptive metadata --
        never used to change which champion is selected.
        """
        n_windows = n_windows or cls._VALIDATION_REPORT_WINDOWS
        n = len(x_df)
        val_start = n - max(1, int(n * val_fraction))
        val_df = x_df.iloc[val_start:]
        if val_df.empty:
            return []
        chunk_size = max(1, len(val_df) // n_windows)
        windows = []
        for i in range(n_windows):
            start = i * chunk_size
            end = len(val_df) if i == n_windows - 1 else (i + 1) * chunk_size
            if start >= len(val_df):
                break
            windows.append(cls._context_snapshot(val_df.iloc[start:end]))
        return windows

    @classmethod
    def _windowed_metric_report(
        cls, y_true, y_pred, is_classification: bool, target_type: str,
        y_scaler, n_windows: int | None = None,
    ) -> list[dict]:
        """Per-window metric over the SAME (already chronologically-ordered)
        validation predictions used for the aggregate metric -- same metric
        family as the aggregate (accuracy for classification, real-unit MSE
        for regression via _unscale_mse), just sliced into contiguous
        chronological chunks instead of one number for the whole period.

        Window boundaries here are independent of _context_windows' (the
        validation set for sequence models has `window - 1` fewer rows than
        the flat feature set, so exact row-for-row alignment between the
        two isn't attempted) -- both slice "the same relative portion of
        the validation period" into n_windows, which is precise enough to
        see whether performance tracks market conditions without requiring
        surgical index-matching across flat and sequence representations.
        """
        import numpy as np
        from sklearn.metrics import accuracy_score, mean_squared_error, roc_auc_score

        n_windows = n_windows or cls._VALIDATION_REPORT_WINDOWS
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        n = len(y_true)
        if n == 0:
            return []
        chunk_size = max(1, n // n_windows)
        report = []
        for i in range(n_windows):
            start = i * chunk_size
            end = n if i == n_windows - 1 else (i + 1) * chunk_size
            if start >= n:
                break
            yt, yp = y_true[start:end], y_pred[start:end]
            if len(yt) == 0:
                continue
            entry = {"n_samples": int(len(yt))}
            if is_classification:
                yt_int = yt.astype(int)
                yp_labels = yp if yp.dtype.kind in "iu" or set(np.unique(yp)) <= {0, 1, 2} else (yp >= 0.5).astype(int)
                entry["accuracy"] = float(accuracy_score(yt_int, yp_labels))
                if target_type != CLASSIFICATION_MULTICLASS_TYPE and len(np.unique(yt_int)) == 2:
                    try:
                        entry["auc"] = float(roc_auc_score(yt_int, yp))
                    except ValueError:
                        pass  # single-class window can't score AUC
            else:
                entry["mse"] = cls._unscale_mse(mean_squared_error(yt, yp), y_scaler)
            report.append(entry)
        return report

    def _keras_final_layer(self, is_classification: bool, target_type: str, num_classes: int):
        import tensorflow as tf
        if target_type == CLASSIFICATION_MULTICLASS_TYPE:
            return tf.keras.layers.Dense(num_classes, activation='softmax')
        if is_classification:
            return tf.keras.layers.Dense(1, activation='sigmoid')
        return tf.keras.layers.Dense(1)

    def _keras_compile_kwargs(self, is_classification: bool, target_type: str) -> dict:
        if target_type == CLASSIFICATION_MULTICLASS_TYPE:
            return {'loss': 'sparse_categorical_crossentropy', 'metrics': ['accuracy']}
        if is_classification:
            import tensorflow as tf
            return {'loss': 'binary_crossentropy', 'metrics': ['accuracy', tf.keras.metrics.AUC(name='auc')]}
        return {'loss': 'mse', 'metrics': []}

    @staticmethod
    def _label_dtype(is_classification: bool, target_type: str):
        # sparse_categorical_crossentropy requires integer class indices;
        # binary_crossentropy and mse both accept float32.
        return np.int32 if target_type == CLASSIFICATION_MULTICLASS_TYPE else np.float32

    @staticmethod
    def _keras_val_metrics(history, is_classification: bool) -> dict:
        """Held-out validation metrics only -- never the training-set loss."""
        metrics = {'val_loss': float(history.history['val_loss'][-1])}
        if is_classification:
            if 'val_accuracy' in history.history:
                metrics['val_accuracy'] = float(history.history['val_accuracy'][-1])
            if 'val_auc' in history.history:
                metrics['val_auc'] = float(history.history['val_auc'][-1])
        return metrics

    @staticmethod
    def _keras_windowed_predictions(model, x_val, is_classification: bool, target_type: str):
        """Predictions shaped for _windowed_metric_report: probability of
        class 1 for binary (sigmoid output is already that), hard label for
        multiclass (argmax over the softmax output), raw value for
        regression. Keras' own history object only has the ONE aggregate
        val_loss/val_accuracy/val_auc for the whole split -- an explicit
        predict() call is needed to slice performance into windows."""
        raw = model.predict(x_val, verbose=0)
        if target_type == CLASSIFICATION_MULTICLASS_TYPE:
            return np.argmax(raw, axis=1)
        return raw.reshape(-1)

    @staticmethod
    def _unscale_mse(mse_scaled: float, y_scaler) -> float:
        """MSE computed on a StandardScaler-scaled target, converted back to
        the target's real units: Var(scaled)=1 => MSE_real = MSE_scaled *
        scale_^2. Regression-only; never called for classification targets
        (which are never scaled in the first place)."""
        if y_scaler is None:
            return float(mse_scaled)
        return float(mse_scaled * (y_scaler.scale_[0] ** 2))

    def _train_mlp_model(self, x_train, y_train, ticker, target_col, *, is_classification, y_scaler, context_windows, model_path):
        """Тренування MLP моделі"""
        import joblib
        from sklearn.metrics import accuracy_score, mean_squared_error, roc_auc_score
        from sklearn.neural_network import MLPClassifier, MLPRegressor

        target_type = self.config_loader.target_type_for(target_col)

        # Chronological split (flat single-row samples -> no window overlap to purge)
        x_train_split, x_val, y_train_split, y_val = self._chronological_split(x_train, y_train)

        model_cls = MLPClassifier if is_classification else MLPRegressor
        model = model_cls(
            hidden_layer_sizes=(128, 64),
            max_iter=self.config_loader.REDUCED_EPOCHS,
            random_state=42,
            verbose=0
        )
        model.fit(x_train_split, y_train_split.astype(int) if is_classification else y_train_split)

        # Save model
        joblib.dump(model, model_path)

        if is_classification:
            y_pred = model.predict(x_val)
            accuracy = accuracy_score(y_val.astype(int), y_pred)
            metrics = {'accuracy': float(accuracy)}
            windowed_pred = y_pred  # hard labels -- correct as-is for multiclass windowed accuracy
            if len(np.unique(y_val)) == 2:
                try:
                    proba = model.predict_proba(x_val)[:, 1]
                    metrics['auc'] = float(roc_auc_score(y_val.astype(int), proba))
                    windowed_pred = proba  # probabilities, so windowed AUC is meaningful too
                except ValueError:
                    pass  # a validation split with a single class can't score AUC
            print(f"    🎯 MLP - Accuracy: {accuracy:.4f} - збережено: {model_path.name}")
        else:
            y_pred = model.predict(x_val)
            mse = self._unscale_mse(mean_squared_error(y_val, y_pred), y_scaler)
            metrics = {'mse': mse}
            windowed_pred = y_pred
            print(f"    🎯 MLP - MSE: {mse:.6f} - збережено: {model_path.name}")

        metrics['validation_windows'] = self._windowed_metric_report(
            y_val, windowed_pred, is_classification, target_type, y_scaler
        )
        metrics['context_windows'] = context_windows

        self._log_mlflow_run(
            ticker, target_col, "mlp",
            params={"hidden_layers": "128,64", "max_iter": self.config_loader.REDUCED_EPOCHS},
            metrics={k: v for k, v in metrics.items() if k not in ('validation_windows', 'context_windows')},
            artifact_path=str(model_path)
        )
        return metrics

    def _train_cnn_model(self, x_train, y_train, ticker, target_col, *, is_classification, y_scaler, context_windows, model_path):
        """Тренування CNN моделі. x_train: (n, window, n_features) real
        chronological sequences (see _build_sequences) -- Conv1D applies
        genuine temporal kernels across `window` consecutive trading days,
        not an arbitrary axis over one flattened feature vector."""
        import tensorflow as tf

        target_type = self.config_loader.target_type_for(target_col)
        num_classes = int(len(np.unique(y_train)))
        label_dtype = self._label_dtype(is_classification, target_type)
        window, num_features = x_train.shape[1], x_train.shape[2]

        # Chronological split, purging (window - 1) rows at the boundary so
        # no validation window shares days with the last training window.
        x_train_split, x_val, y_train_split, y_val = self._chronological_split(
            x_train, y_train, purge=window - 1
        )
        y_train_split = y_train_split.astype(label_dtype)
        y_val = y_val.astype(label_dtype)

        # Create CNN model
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(window, num_features)),
            tf.keras.layers.Conv1D(32, 3, activation='relu'),
            tf.keras.layers.MaxPooling1D(2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            self._keras_final_layer(is_classification, target_type, num_classes)
        ])

        model.compile(optimizer='adam', **self._keras_compile_kwargs(is_classification, target_type))

        # Train
        epochs = self.config_loader.REDUCED_EPOCHS or 10
        history = model.fit(
            x_train_split, y_train_split,
            epochs=epochs,
            validation_data=(x_val, y_val),
            verbose=0
        )

        # Save model
        model.save(model_path)

        metrics = self._keras_val_metrics(history, is_classification)
        if not is_classification:
            metrics['mse'] = self._unscale_mse(metrics.pop('val_loss'), y_scaler)

        windowed_pred = self._keras_windowed_predictions(model, x_val, is_classification, target_type)
        metrics['validation_windows'] = self._windowed_metric_report(
            y_val, windowed_pred, is_classification, target_type, y_scaler
        )
        metrics['context_windows'] = context_windows

        # Log to MLflow
        self._log_mlflow_run(
            ticker, target_col, "cnn",
            params={"conv_filters": 32, "kernel_size": 3, "epochs": epochs, "window": window},
            metrics={k: v for k, v in metrics.items() if k not in ('validation_windows', 'context_windows')},
            artifact_path=str(model_path)
        )

        print(f"    🎯 CNN - {metrics} - збережено: {model_path.name}")
        return metrics

    def _train_lstm_model(self, x_train, y_train, ticker, target_col, *, is_classification, y_scaler, context_windows, model_path):
        """Тренування LSTM моделі. x_train: (n, window, n_features) real
        chronological sequences -- LSTM recurrence now runs across `window`
        real trading days instead of a fake sequence length of 1."""
        import tensorflow as tf

        target_type = self.config_loader.target_type_for(target_col)
        num_classes = int(len(np.unique(y_train)))
        label_dtype = self._label_dtype(is_classification, target_type)
        window, num_features = x_train.shape[1], x_train.shape[2]

        x_train_split, x_val, y_train_split, y_val = self._chronological_split(
            x_train, y_train, purge=window - 1
        )
        y_train_split = y_train_split.astype(label_dtype)
        y_val = y_val.astype(label_dtype)

        # Create LSTM model
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(window, num_features)),
            tf.keras.layers.LSTM(64),
            tf.keras.layers.Dense(32, activation='relu'),
            self._keras_final_layer(is_classification, target_type, num_classes)
        ])

        model.compile(optimizer='adam', **self._keras_compile_kwargs(is_classification, target_type))

        # Train
        epochs = self.config_loader.REDUCED_EPOCHS or 10
        history = model.fit(
            x_train_split, y_train_split,
            epochs=epochs,
            validation_data=(x_val, y_val),
            verbose=0
        )

        # Save model
        model.save(model_path)

        metrics = self._keras_val_metrics(history, is_classification)
        if not is_classification:
            metrics['mse'] = self._unscale_mse(metrics.pop('val_loss'), y_scaler)

        windowed_pred = self._keras_windowed_predictions(model, x_val, is_classification, target_type)
        metrics['validation_windows'] = self._windowed_metric_report(
            y_val, windowed_pred, is_classification, target_type, y_scaler
        )
        metrics['context_windows'] = context_windows

        # Log to MLflow
        self._log_mlflow_run(
            ticker, target_col, "lstm",
            params={"units": 64, "epochs": epochs, "window": window},
            metrics={k: v for k, v in metrics.items() if k not in ('validation_windows', 'context_windows')},
            artifact_path=str(model_path)
        )

        print(f"    🎯 LSTM - {metrics} - збережено: {model_path.name}")
        return metrics

    def _train_gru_model(self, x_train, y_train, ticker, target_col, *, is_classification, y_scaler, context_windows, model_path):
        """Тренування GRU моделі. x_train: (n, window, n_features) real
        chronological sequences."""
        import tensorflow as tf

        target_type = self.config_loader.target_type_for(target_col)
        num_classes = int(len(np.unique(y_train)))
        label_dtype = self._label_dtype(is_classification, target_type)
        window, num_features = x_train.shape[1], x_train.shape[2]

        x_train_split, x_val, y_train_split, y_val = self._chronological_split(
            x_train, y_train, purge=window - 1
        )
        y_train_split = y_train_split.astype(label_dtype)
        y_val = y_val.astype(label_dtype)

        # Create GRU model
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(window, num_features)),
            tf.keras.layers.GRU(64),
            tf.keras.layers.Dense(32, activation='relu'),
            self._keras_final_layer(is_classification, target_type, num_classes)
        ])

        model.compile(optimizer='adam', **self._keras_compile_kwargs(is_classification, target_type))

        # Train
        epochs = self.config_loader.REDUCED_EPOCHS or 10
        history = model.fit(
            x_train_split, y_train_split,
            epochs=epochs,
            validation_data=(x_val, y_val),
            verbose=0
        )

        # Save model
        model.save(model_path)

        metrics = self._keras_val_metrics(history, is_classification)
        if not is_classification:
            metrics['mse'] = self._unscale_mse(metrics.pop('val_loss'), y_scaler)

        windowed_pred = self._keras_windowed_predictions(model, x_val, is_classification, target_type)
        metrics['validation_windows'] = self._windowed_metric_report(
            y_val, windowed_pred, is_classification, target_type, y_scaler
        )
        metrics['context_windows'] = context_windows

        # Log to MLflow
        self._log_mlflow_run(
            ticker, target_col, "gru",
            params={"units": 64, "epochs": epochs, "window": window},
            metrics={k: v for k, v in metrics.items() if k not in ('validation_windows', 'context_windows')},
            artifact_path=str(model_path)
        )

        print(f"    🎯 GRU - {metrics} - збережено: {model_path.name}")
        return metrics

    def _train_transformer_model(self, x_train, y_train, ticker, target_col, *, is_classification, y_scaler, context_windows, model_path):
        """Тренування Transformer моделі. x_train: (n, window, n_features)
        real chronological sequences -- MultiHeadAttention now has `window`
        real trading days to attend over instead of a single timestep,
        where self-attention over one position is a no-op."""
        import tensorflow as tf

        target_type = self.config_loader.target_type_for(target_col)
        num_classes = int(len(np.unique(y_train)))
        label_dtype = self._label_dtype(is_classification, target_type)
        window, num_features = x_train.shape[1], x_train.shape[2]

        x_train_split, x_val, y_train_split, y_val = self._chronological_split(
            x_train, y_train, purge=window - 1
        )
        y_train_split = y_train_split.astype(label_dtype)
        y_val = y_val.astype(label_dtype)

        # Create Transformer model (functional API for better stability with MultiHeadAttention)
        inputs = tf.keras.layers.Input(shape=(window, num_features))
        attention = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=32)(inputs, inputs)
        pooling = tf.keras.layers.GlobalAveragePooling1D()(attention)
        dense1 = tf.keras.layers.Dense(64, activation='relu')(pooling)
        outputs = self._keras_final_layer(is_classification, target_type, num_classes)(dense1)

        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        model.compile(optimizer='adam', **self._keras_compile_kwargs(is_classification, target_type))

        # Train
        epochs = self.config_loader.REDUCED_EPOCHS or 10
        history = model.fit(
            x_train_split, y_train_split,
            epochs=epochs,
            validation_data=(x_val, y_val),
            verbose=0
        )

        # Save model
        model.save(model_path)

        metrics = self._keras_val_metrics(history, is_classification)
        if not is_classification:
            metrics['mse'] = self._unscale_mse(metrics.pop('val_loss'), y_scaler)

        windowed_pred = self._keras_windowed_predictions(model, x_val, is_classification, target_type)
        metrics['validation_windows'] = self._windowed_metric_report(
            y_val, windowed_pred, is_classification, target_type, y_scaler
        )
        metrics['context_windows'] = context_windows

        # Log to MLflow
        self._log_mlflow_run(
            ticker, target_col, "transformer",
            params={"head_size": 128, "num_heads": 4, "epochs": epochs, "window": window},
            metrics={k: v for k, v in metrics.items() if k not in ('validation_windows', 'context_windows')},
            artifact_path=str(model_path)
        )

        print(f"    🎯 Transformer - {metrics} - збережено: {model_path.name}")
        return metrics

    @staticmethod
    def _save_tabnet(model, model_path):
        """TabNet appends the extension itself; hand it the stem.

        pytorch_tabnet's save_model calls shutil.make_archive(path, "zip"),
        which writes path + ".zip". Given a path already ending in .zip it
        produces model_..._tabnet.zip.zip -- and the skip check, the sidecar
        and Stage 5 all look for the single-suffix name. So every TabNet
        model was written where nothing would look for it, retrained on
        every run, and its metrics filed against a file that did not exist.
        """
        stem = str(model_path)
        if stem.endswith(".zip"):
            stem = stem[: -len(".zip")]
        model.save_model(stem)

    def _train_tabnet_model(self, x_train, y_train, ticker, target_col, *, is_classification, y_scaler, context_windows, model_path):
        """Тренування TabNet моделі"""
        try:
            import torch
            from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor
            from sklearn.metrics import accuracy_score, mean_squared_error, roc_auc_score

            target_type = self.config_loader.target_type_for(target_col)

            # Chronological split (flat single-row samples -> no window overlap to purge)
            x_train_split, x_val, y_train_split, y_val = self._chronological_split(x_train, y_train)

            tabnet_kwargs = dict(
                n_d=64, n_a=64,
                n_steps=3,
                gamma=1.5,
                lambda_sparse=1e-3,
                optimizer_fn=torch.optim.Adam,
                optimizer_params={"lr": 2e-2},
                mask_type='entmax',
                scheduler_params={"step_size": 10, "gamma": 0.9},
                verbose=0
            )
            max_epochs = self.config_loader.REDUCED_EPOCHS or 20

            if is_classification:
                model = TabNetClassifier(**tabnet_kwargs)
                y_train_labels = y_train_split.values.astype(np.int64)
                y_val_labels = y_val.values.astype(np.int64)
                model.fit(
                    X_train=x_train_split.values,
                    y_train=y_train_labels,
                    eval_set=[(x_val.values, y_val_labels)],
                    max_epochs=max_epochs,
                    patience=5,
                    batch_size=1024,
                    virtual_batch_size=128,
                    num_workers=0,
                    drop_last=False
                )
                self._save_tabnet(model, model_path)

                y_pred = model.predict(x_val.values)
                accuracy = accuracy_score(y_val_labels, y_pred)
                metrics = {'accuracy': float(accuracy)}
                windowed_pred = y_pred
                if len(np.unique(y_val_labels)) == 2:
                    try:
                        proba = model.predict_proba(x_val.values)[:, 1]
                        metrics['auc'] = float(roc_auc_score(y_val_labels, proba))
                        windowed_pred = proba
                    except ValueError:
                        pass
                print(f"    🎯 TabNet - Accuracy: {accuracy:.4f} - збережено: {model_path.name}")
            else:
                model = TabNetRegressor(**tabnet_kwargs)
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
                self._save_tabnet(model, model_path)

                y_pred = model.predict(x_val.values).reshape(-1)
                mse = self._unscale_mse(mean_squared_error(y_val, y_pred), y_scaler)
                metrics = {'mse': mse}
                windowed_pred = y_pred
                print(f"    🎯 TabNet - MSE: {mse:.6f} - збережено: {model_path.name}")

            metrics['validation_windows'] = self._windowed_metric_report(
                y_val_labels if is_classification else y_val, windowed_pred,
                is_classification, target_type, y_scaler
            )
            metrics['context_windows'] = context_windows
            return metrics

        except ImportError:
            print("    ⚠️ TabNet не встановлено, пропускаємо")
            return {'error': 'pytorch_tabnet not installed'}

    def _train_autoencoder_model(self, x_train, y_train, ticker, target_col, *, is_classification, y_scaler, context_windows, model_path):
        """Тренування Autoencoder моделі (encoder + supervised head, not a
        reconstruction autoencoder despite the name -- kept as-is)."""
        import tensorflow as tf

        target_type = self.config_loader.target_type_for(target_col)
        num_classes = int(pd.Series(y_train).nunique())
        label_dtype = self._label_dtype(is_classification, target_type)

        # Chronological split (flat single-row samples -> no window overlap to purge)
        x_train_split, x_val, y_train_split, y_val = self._chronological_split(x_train, y_train)

        # Ensure numpy float32
        x_train_np = x_train_split.values.astype(np.float32)
        x_val_np = x_val.values.astype(np.float32)
        y_train_np = y_train_split.values.astype(label_dtype)
        y_val_np = y_val.values.astype(label_dtype)

        # Create autoencoder model
        input_dim = x_train.shape[1]
        encoding_dim = 32

        # Encoder
        input_layer = tf.keras.layers.Input(shape=(input_dim,))
        encoder = tf.keras.layers.Dense(encoding_dim, activation='relu')(input_layer)

        # Supervised head
        head = tf.keras.layers.Dense(64, activation='relu')(encoder)
        output = self._keras_final_layer(is_classification, target_type, num_classes)(head)

        # Create model
        model = tf.keras.Model(inputs=input_layer, outputs=output)

        model.compile(optimizer='adam', **self._keras_compile_kwargs(is_classification, target_type))

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
        model.save(model_path)

        metrics = self._keras_val_metrics(history, is_classification)
        if not is_classification:
            metrics['mse'] = self._unscale_mse(metrics.pop('val_loss'), y_scaler)

        windowed_pred = self._keras_windowed_predictions(model, x_val_np, is_classification, target_type)
        metrics['validation_windows'] = self._windowed_metric_report(
            y_val_np, windowed_pred, is_classification, target_type, y_scaler
        )
        metrics['context_windows'] = context_windows

        print(f"    🎯 Autoencoder - {metrics} - збережено: {model_path.name}")
        return metrics

    def _batch_fingerprint(self) -> str:
        """Which batch this run is training on.

        prepare writes raw_db_fingerprint.json beside the parquets, holding a
        hash of the raw data and a hash of the code that built the features.
        Together they identify the batch: change either and the features are
        different numbers, so a model fitted to the old ones is stale.

        Empty when the file is absent -- an unidentifiable batch disables the
        reuse check rather than silently reusing against nothing.
        """
        cached = getattr(self, "_batch_fp_cache", None)
        if cached is not None:
            return cached
        value = ""
        try:
            path = self.path_manager.batch_dir / "raw_db_fingerprint.json"
            if path.exists():
                payload = json.loads(path.read_text(encoding="utf-8"))
                value = f"{payload.get('fingerprint','')[:16]}:{payload.get('code_fingerprint','')[:16]}"
        except (OSError, json.JSONDecodeError) as exc:
            self.logger.warning("Could not read the batch fingerprint: %s", exc)
        self._batch_fp_cache = value
        return value

    def _sidecar_path(self, model_filename: str) -> Path:
        return self.path_manager.batch_dir / f"{model_filename}.metrics.json"

    def _save_sidecar(self, model_filename: str, metrics: dict, selected_features: list) -> None:
        """Write a model's metrics next to the model itself.

        A trained model's numbers otherwise live only in the results file of
        the run that produced them. The next run finds the model file on
        disk, skips training, and has nothing left to report -- which is how
        1,405 heavy models came to carry {"info": "already_exists"} and no
        measured quality whatsoever.
        """
        try:
            payload = {
                "metrics": metrics,
                "selected_features": list(selected_features or []),
                "saved_at": datetime.now().isoformat(),
                # Which batch this fit belongs to. Without it, "the model
                # file exists" is the only reuse test there is, and a model
                # fitted to last week's features is indistinguishable from
                # one fitted to today's.
                "batch_fingerprint": self._batch_fingerprint(),
            }
            self._sidecar_path(model_filename).write_text(
                json.dumps(payload, ensure_ascii=False), encoding="utf-8"
            )
        except Exception as exc:
            # Never fail a completed training run over its bookkeeping.
            self.logger.warning("Could not save metrics sidecar for %s: %s",
                                model_filename, exc)

    def _load_sidecar(self, model_filename: str) -> dict | None:
        """The saved metrics for a model, or None when there are none.

        None means "this model predates the sidecar, or its file was lost" --
        the caller then falls back to the old behaviour rather than inventing
        numbers for a model it did not measure.
        """
        path = self._sidecar_path(model_filename)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            self.logger.warning("Unreadable metrics sidecar %s: %s", path, exc)
            return None
        return payload if isinstance(payload, dict) and payload.get("metrics") else None

    def _get_model_max_features(self, model_type):
        """Feature budget for `model_type`, from the project config.

        This used to be a hardcoded map (mlp 256, cnn 64, lstm/gru/transformer/
        autoencoder 128, tabnet 256) and it was the copy that actually ran:
        measured across 4,613 trained artifacts, every heavy model's feature
        count matched it exactly and matched src/config/models.yaml on none of
        the seven types. The configured numbers had never taken effect
        anywhere, because the local trainer passed them to the model
        constructor as a hyperparameter instead of using them as a budget.

        One source now: src/config/feature_budget. The fallback keeps this
        script working if it is ever run without the repo's config on the
        path, which is the situation the hardcoded map was really guarding
        against.
        """
        try:
            from src.config.feature_budget import get_model_max_features
            return get_model_max_features(model_type)
        except ImportError:
            # Mirrors feature_budget.DEFAULT_MAX_FEATURES; kept as a literal
            # precisely because this branch is the one where that module
            # could not be imported.
            return 35

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    print("🔥 COLAB TRAINING CONTROLLER - CLEAN VERSION")
    print("="*80)

    # Initialize controller first to setup paths
    controller = ColabTrainingController()

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