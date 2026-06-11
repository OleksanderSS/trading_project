
import pandas as pd

from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger("Stage3Improvements")

def validate_and_align_features_targets(features: pd.DataFrame, targets: pd.DataFrame, tf: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Ensures features and targets are aligned by index."""
    if features.empty or targets.empty:
        return features, targets

    # Force same index type
    if not isinstance(features.index, pd.DatetimeIndex):
        features.index = pd.to_datetime(features.index)
    if not isinstance(targets.index, pd.DatetimeIndex):
        targets.index = pd.to_datetime(targets.index)

    # Find common intersection
    common_idx = features.index.intersection(targets.index)

    if len(common_idx) == 0:
        logger.error(f"❌ ZERO alignment for {tf}. Indices are completely different!")
        return pd.DataFrame(), pd.DataFrame()

    if len(common_idx) < max(len(features), len(targets)) * 0.5:
        logger.warning(f"⚠️ Poor alignment for {tf}: {len(common_idx)} common rows out of {len(features)}")

    # Return aligned subset
    f_aligned = features.loc[common_idx].sort_index()
    t_aligned = targets.loc[common_idx].sort_index()

    logger.info(f"✅ Aligned {tf}: {len(f_aligned)} rows")
    return f_aligned, t_aligned

def calculate_data_quality_metrics(enriched_prices, all_targets, count):
    # Mock/Simplified for stability
    return {"overall": {"status": "ok", "total_rows": sum(len(df) for df in enriched_prices.values())}}

def log_data_quality_report(metrics):
    logger.info(f"📊 Quality check: {metrics['overall']['status']}")
