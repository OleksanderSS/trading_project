# Diagnostic Whitelist
# This file contains patterns that should be ignored by the diagnostic auditor
# to reduce false positives in risk detection.

# Safe target-related parameters that are NOT data leakage
SAFE_PATTERNS = [
    "target_volatility",
    "target_return",
    "target_risk",
    "target_col",
    "target_series",
    "target_bits",
    "target_name",
    "target_prefix",
    "target_cols",
    "target_asset",
    "target_version",
    "target_return_1d",
    "target_risk_pct",
    # Financial metrics and weights
    "target_allocations",
    "target_value",
    "target_weight",
    "target_return",
    # Model configuration
    "target_type",
    "target_count",
    "target_stationary",
    # File paths/Checkpoints
    "checkpoint_",
]
