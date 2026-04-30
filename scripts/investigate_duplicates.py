#!/usr/bin/env python3
"""
Deep investigation into duplicate data issues.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def _load_data():
    """Load features and targets data"""
    db_path = Path("data/colab/accumulated/test_ticker_amd_target_return_1d")
    df_features = pd.read_parquet(db_path / "features.parquet")
    df_targets = pd.read_parquet(db_path / "targets.parquet")
    return df_features, df_targets

def _investigate_features_duplicates(df_features):
    """Investigate duplicates in features data"""
    logger.info("\n📊 FEATURES INVESTIGATION")
    logger.info("-" * 80)
    
    # Check if index is duplicated
    logger.info(f"Index duplicates: {df_features.index.duplicated().sum()}")
    
    # Find duplicate rows
    dup_mask = df_features.duplicated(keep=False)
    dup_rows = df_features[dup_mask]
    
    logger.info(f"Total duplicate rows: {dup_rows.shape[0]}")
    logger.info(f"Unique duplicate groups: {dup_rows.duplicated().sum()}")
    
    # Sample duplicates
    if len(dup_rows) > 0:
        logger.info("\nSample duplicate rows:")
        sample_dup = dup_rows.iloc[:2]
        logger.info(f"  Rows are identical: {sample_dup.iloc[0].equals(sample_dup.iloc[1])}")
        
        # Check which columns differ
        if len(sample_dup) >= 2:
            diff_cols = []
            for col in sample_dup.columns:
                if sample_dup.iloc[0][col] != sample_dup.iloc[1][col]:
                    diff_cols.append(col)
            
            if diff_cols:
                logger.info(f"  Columns that differ: {diff_cols}")
            else:
                logger.info("  All columns are identical")
    
    # Check for patterns in duplicates
    logger.info("\nDuplicate patterns:")
    dup_counts = df_features.duplicated(keep=False).groupby(df_features.index).sum()
    logger.info(f"  Max duplicates per index: {dup_counts.max()}")
    logger.info(f"  Mean duplicates per index: {dup_counts.mean():.2f}")
    
    return {
        "total_duplicates": int(df_features.duplicated().sum()),
        "duplicate_percentage": float(df_features.duplicated().sum() / len(df_features) * 100),
        "unique_rows": len(df_features.drop_duplicates())
    }

def _investigate_targets_duplicates(df_targets):
    """Investigate duplicates in targets data"""
    logger.info("\nTarget shape: {}".format(df_targets.shape))
    logger.info("Target columns: {}".format(list(df_targets.columns)))
    
    # Check target values
    for col in df_targets.columns:
        logger.info("\n{}:".format(col))
        logger.info("  Unique values: {}".format(df_targets[col].nunique()))
        logger.info("  Value counts:\n{}".format(df_targets[col].value_counts()))
        
        # Only compute stats for numeric columns
        if pd.api.types.is_numeric_dtype(df_targets[col]):
            logger.info("  Min: {}, Max: {}".format(df_targets[col].min(), df_targets[col].max()))
            logger.info("  Mean: {:.4f}, Std: {:.4f}".format(df_targets[col].mean(), df_targets[col].std()))
        else:
            logger.info("  Type: {} (non-numeric)".format(df_targets[col].dtype))
    
    # Check duplicates
    dup_mask_targets = df_targets.duplicated(keep=False)
    logger.info("\nTotal duplicate rows: {}".format(dup_mask_targets.sum()))
    logger.info("Unique rows: {}".format(len(df_targets.drop_duplicates())))
    
    # Check if all targets are the same
    if df_targets.shape[1] > 0:
        first_col = df_targets.columns[0]
        all_same = (df_targets[first_col] == df_targets[first_col].iloc[0]).all()
        logger.info("All target values identical: {}".format(all_same))
    
    investigation = {
        "total_duplicates": int(df_targets.duplicated().sum()),
        "duplicate_percentage": float(df_targets.duplicated().sum() / len(df_targets) * 100),
        "unique_rows": len(df_targets.drop_duplicates()),
        "columns": list(df_targets.columns),
        "value_distribution": {}
    }
    
    for col in df_targets.columns:
        investigation["value_distribution"][col] = df_targets[col].value_counts().to_dict()
    
    return investigation

def _investigate_correlation(df_features, df_targets):
    """Investigate correlation between features and targets duplicates"""
    logger.info("\nCORRELATION INVESTIGATION")
    logger.info("-" * 80)
    
    # Check if features and targets have same duplicates
    feat_dup_indices = set(df_features[df_features.duplicated(keep=False)].index)
    targ_dup_indices = set(df_targets[df_targets.duplicated(keep=False)].index)
    
    overlap = feat_dup_indices & targ_dup_indices
    logger.info("Duplicate indices in both: {}".format(len(overlap)))
    logger.info("Duplicate indices only in features: {}".format(len(feat_dup_indices - targ_dup_indices)))
    logger.info("Duplicate indices only in targets: {}".format(len(targ_dup_indices - feat_dup_indices)))
    
    return {
        "overlap_duplicates": len(overlap),
        "features_only": len(feat_dup_indices - targ_dup_indices),
        "targets_only": len(targ_dup_indices - feat_dup_indices)
    }

def investigate_duplicates():
    """Investigate duplicate data in detail"""
    logger.info("=" * 80)
    logger.info("INVESTIGATING DUPLICATES")
    logger.info("=" * 80)
    
    # Load data
    df_features, df_targets = _load_data()
    
    investigation = {
        "features": {},
        "targets": {},
        "correlation": {}
    }
    
    # ===== FEATURES INVESTIGATION =====
    investigation["features"] = _investigate_features_duplicates(df_features)
    
    # ===== TARGETS INVESTIGATION =====
    investigation["targets"] = _investigate_targets_duplicates(df_targets)
    
    # ===== CORRELATION INVESTIGATION =====
    investigation["correlation"] = _investigate_correlation(df_features, df_targets)
    
    # ===== SAVE INVESTIGATION =====
    report_path = Path("results/duplicate_investigation.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_path, 'w') as f:
        json.dump(investigation, f, indent=2)
    
    logger.info(f"\n✅ Investigation saved to: {report_path}")
    
    # ===== RECOMMENDATIONS =====
    logger.info("\n" + "=" * 80)
    logger.info("RECOMMENDATIONS")
    logger.info("=" * 80)
    
    if investigation["targets"]["duplicate_percentage"] > 90:
        logger.warning("🔴 CRITICAL: Targets are almost entirely duplicated")
        logger.warning("   Action: Check target calculation in pipeline")
    
    if investigation["features"]["duplicate_percentage"] > 50:
        logger.warning("🔴 CRITICAL: Features are heavily duplicated")
        logger.warning("   Action: Check feature merging logic")
    
    logger.info("\nSuggested fixes:")
    logger.info("1. Remove duplicates: df.drop_duplicates(inplace=True)")
    logger.info("2. Check target calculation: verify shift parameter")
    logger.info("3. Check feature merging: verify no broadcast issues")
    logger.info("4. Add deduplication step to pipeline")

if __name__ == "__main__":
    investigate_duplicates()
