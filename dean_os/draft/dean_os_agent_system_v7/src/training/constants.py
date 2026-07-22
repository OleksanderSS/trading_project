"""
Training Constants
This module contains all magic numbers and constants used across the training system.
Centralization ensures ease of modification and consistency.
"""

from src.factories.model_factory import ModelFactory

# ============================================================================
# Batch Training Constants
# ============================================================================

# Default batch configuration values
BATCH_TRAINER_DEFAULT_BATCH_SIZE = 10
"""Default number of tickers to train in parallel (default batch size)"""

BATCH_TRAINER_DEFAULT_MAX_MEMORY_GB = 12.0
"""Maximum memory allowed per batch training process in GB"""


# ============================================================================
# Progressive Training Constants
# ============================================================================

# Batch size progression
PROGRESSIVE_INITIAL_BATCH_SIZE = 5
"""Starting batch size for progressive training"""

PROGRESSIVE_MAX_BATCH_SIZE = 20
"""Maximum batch size limit for progressive training"""

PROGRESSIVE_BATCH_GROWTH_FACTOR = 1.5
"""Factor to multiply batch size when scaling up (e.g., 5 * 1.5 = 7.5)"""


# Quality Thresholds
PROGRESSIVE_MIN_ACCURACY_THRESHOLD = 0.75
"""Minimum model accuracy threshold (0.0 to 1.0)"""

PROGRESSIVE_MAX_LOSS_THRESHOLD = 0.5
"""Maximum loss threshold (model fails if loss exceeds this)"""


# Checkpointing
PROGRESSIVE_CHECKPOINT_INTERVAL = 3
"""Save checkpoint every N batches"""

PROGRESSIVE_MAX_TIME_HOURS = 10.0
"""Maximum wall-clock time allowed for progressive training"""

PROGRESSIVE_MAX_MEMORY_GB = 8.0
"""Maximum memory allowed for progressive training"""


# ============================================================================
# Modeling Stage Constants
# ============================================================================

# Test/Train split
DEFAULT_TEST_SIZE = 0.2
"""Default test/validation set fraction (0.0 to 1.0)"""


# ============================================================================
# Path Configuration Defaults
# ============================================================================

# Default directory paths (used as fallbacks in config)
DEFAULT_MODELS_PATH = "data/trained_models"
"""Default directory for storing trained models"""

DEFAULT_DIARY_PATH = "logs/experience_diary.csv"
"""Default path for model training diary/log CSV"""

DEFAULT_CACHE_PATH = "data/cache"
"""Default directory for caching feature data"""

DEFAULT_SELECTED_FEATURES_CACHE = "data/cache/selected_features.json"
"""Default path for cached selected features file"""

DEFAULT_ACCUMULATION_OUTPUT_DIR = "data/colab/accumulated"
"""Default directory for accumulation output files"""


# ============================================================================
# Feature Selection Constants
# ============================================================================

# Feature selection thresholds
MIN_FEATURE_IMPORTANCE_THRESHOLD = 0.001
"""Minimum feature importance score to retain a feature"""

MAX_FEATURES_TO_SELECT = 100
"""Maximum number of features to select (prevents overfitting)"""

MIN_FEATURES_TO_KEEP = 5
"""Minimum number of features to keep even if below threshold"""


# ============================================================================
# System Configuration Constants
# ============================================================================

# Resource limits
COLAB_MAX_MEMORY_GB = 12.0
"""Typical Colab RAM limit in GB"""

LOCAL_MAX_MEMORY_GB = 8.0
"""Typical local machine RAM limit in GB"""

# Parallelization
DEFAULT_N_JOBS_PARALLEL = -1
"""Use all available cores (-1) for parallel jobs when n_tickers > 1"""

DEFAULT_N_JOBS_SEQUENTIAL = 1
"""Use single core for sequential execution"""


# ============================================================================
# Model Configuration Constants
# ============================================================================

# Default enabled model types - use ModelFactory for canonical source
DEFAULT_ENABLED_MODEL_TYPES = ModelFactory.get_available_models()
"""Default list of model types to train"""

# Hyperparameter ranges
LIGHTGBM_DEFAULT_NUM_LEAVES = 31
"""Default number of leaves for LightGBM models"""

RANDOM_FOREST_DEFAULT_N_ESTIMATORS = 100
"""Default number of estimators for Random Forest"""

XGBOOST_DEFAULT_N_ESTIMATORS = 100
"""Default number of trees for XGBoost"""

LINEAR_MODEL_REGULARIZATION_L2 = 0.01
"""Default L2 regularization strength for linear models"""
