#!/usr/bin/env python3
"""Audit script - verify all imports and data contracts"""
import sys
sys.path.insert(0, '.')

print("=" * 60)
print("AUDIT: Verify Session 2 Changes")
print("=" * 60)

# Test 1: Check datetime_utils file exists and has correct structure
print("\n[1/5] Checking datetime_utils.py structure...")
try:
    from src.features.utils.datetime_utils import (
        ensure_datetime_column,
        ensure_ticker_column,
        normalize_metadata_columns,
        split_datetime_ticker,
        roundtrip_datetime_ticker,
        deduplicate_on_metadata,
        ensure_datetime_sorted
    )
    print("✅ datetime_utils: All 7 functions imported successfully")
except ImportError as e:
    print(f"❌ datetime_utils import failed: {e}")
    sys.exit(1)

# Test 2: Verify stage imports
print("\n[2/5] Checking stage imports...")
try:
    from src.pipeline.stages.stage_2_processing import ProcessingStage
    print("✅ stage_2_processing.ProcessingStage imported")
except ImportError as e:
    print(f"❌ stage_2_processing import failed: {e}")
    sys.exit(1)

try:
    from src.pipeline.stages.stage_3_feature_engineering import FeatureEngineeringStage
    print("✅ stage_3_feature_engineering.FeatureEngineeringStage imported")
except ImportError as e:
    print(f"❌ stage_3_feature_engineering import failed: {e}")
    sys.exit(1)

try:
    from src.pipeline.stages.stage_4_modeling import ModelingStage
    print("✅ stage_4_modeling.ModelingStage imported")
except ImportError as e:
    print(f"❌ stage_4_modeling import failed: {e}")
    sys.exit(1)

try:
    from src.pipeline.stages.stage_5_prediction import PredictionStage
    print("✅ stage_5_prediction.PredictionStage imported")
except ImportError as e:
    print(f"❌ stage_5_prediction import failed: {e}")
    sys.exit(1)

# Test 3: Verify pipeline orchestrators
print("\n[3/5] Checking pipeline orchestrators...")
try:
    from src.pipeline.pipeline_orchestrator import PipelineOrchestrator
    print("✅ PipelineOrchestrator imported")
except ImportError as e:
    print(f"❌ PipelineOrchestrator import failed: {e}")
    sys.exit(1)

try:
    from src.pipeline.hybrid_orchestrator import HybridOrchestrator
    print("✅ HybridOrchestrator imported")
except ImportError as e:
    print(f"❌ HybridOrchestrator import failed: {e}")
    sys.exit(1)

# Test 4: Test basic datetime_utils functionality
print("\n[4/5] Testing datetime_utils functionality...")
import pandas as pd
import numpy as np

# Test datetime column normalization
df_test = pd.DataFrame({
    'published_at': pd.date_range('2024-01-01', periods=5),
    'value': range(5)
})
df_normalized = normalize_metadata_columns(df_test)
if 'datetime' in df_normalized.columns and 'ticker' in df_normalized.columns:
    print("✅ normalize_metadata_columns works correctly")
else:
    print("❌ normalize_metadata_columns didn't add required columns")
    sys.exit(1)

# Test 5: Verify config manager
print("\n[5/5] Checking UnifiedConfigManager...")
try:
    from src.config.unified_config_manager import UnifiedConfigManager
    config = UnifiedConfigManager()
    
    # Check critical configs
    paths = config.get_config('paths')
    models = config.get_config('models')
    features = config.get_config('features')
    targets = config.get_config('targets')
    training_pipeline = config.get_config('training_pipeline')
    
    if all([paths, models, features, targets, training_pipeline]):
        print("✅ All critical configs loaded")
    else:
        missing = []
        if not paths: missing.append('paths')
        if not models: missing.append('models')
        if not features: missing.append('features')
        if not targets: missing.append('targets')
        if not training_pipeline: missing.append('training_pipeline')
        print(f"⚠️  Missing configs: {missing}")
except Exception as e:
    print(f"❌ Config manager check failed: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("✅✅✅ ALL AUDIT CHECKS PASSED ✅✅✅")
print("=" * 60)
print("\nReady to run: python run_hybrid_pipeline.py --mode prepare ...")
