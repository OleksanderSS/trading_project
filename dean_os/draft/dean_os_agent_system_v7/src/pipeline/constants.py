"""
Pipeline-wide file name constants — single source of truth.

Import from here instead of re-defining in every module.
"""

# Core data files
FEATURES_FILE = "features.parquet"
TARGETS_FILE = "targets.parquet"

# Metadata files
BATCH_METADATA_FILE = "batch_metadata.json"
COLAB_RESULTS_FILE = "colab_results.json"
LIGHT_MODELS_RESULTS_FILE = "light_models_results.json"
RUNTIME_PARAMS_FILE = "runtime_params.json"

# Glob patterns
SELECTED_FEATURES_PATTERN = "selected_features_*.json"
MODEL_FILES_PATTERN = "model_*"

# Persistent storage directory
PERSISTENT_FEATURES_DIR = "data/processed/features"

# Accumulated Colab data root
COLAB_ACCUMULATED_ROOT = "data/colab/accumulated"
