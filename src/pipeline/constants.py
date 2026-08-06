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


def heavy_model_key(ticker: str, timeframe: str, target: str, model_type: str) -> str:
    """The identity of one heavy (Colab-trained) model.

    The timeframe belongs in the key. It was absent, which was harmless only
    while Colab trained a single model per (ticker, target, model_type) over
    all three timeframes mixed together. Once each timeframe gets its own
    fit, the old key names three different models identically -- and both
    readers assign into models_metadata[key], so the 15m and 60m models were
    overwritten by the 1d one and vanished without a log line.

    Field order matches the light branch's champion key in
    stages/modeling/orchestrator.py, {ticker}_{timeframe}_{target}_{...},
    so both halves of the hybrid read the same way. The Colab writer builds
    the same string in scripts/colab/colab_clean_cell.py::_context_id, and
    the model filename is "model_" + this key + the extension -- which is
    what makes model_resolver's '*{context_id}*' glob find it.
    """
    return f"{ticker}_{timeframe}_{target}_{model_type}"


def model_candidate_filename(ticker: str, timeframe: str, target: str,
                             model_type: str) -> str:
    """Filename for one light (locally trained) model candidate.

    The timeframe is omitted when unknown rather than written as an empty
    field, so a caller that has no timeframe produces the historical name
    instead of "model_AAPL__target_x_mlp" -- a name with a hole in it that
    nothing would parse.
    """
    parts = [p for p in (ticker, timeframe, target, model_type) if p]
    return "model_" + "_".join(parts) + ".joblib"


def champion_filename(ticker: str, timeframe: str, target: str) -> str:
    """Filename for the promoted winner of one (ticker, timeframe, target)."""
    parts = [p for p in (ticker, timeframe, target) if p]
    return "CHAMP_" + "_".join(parts) + ".joblib"
