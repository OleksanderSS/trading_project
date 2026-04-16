# Quick Reference - Pipeline Data Flow & Utilities

## Datetime Utilities Quickstart

### Basic Usage
```python
from src.features.utils.datetime_utils import normalize_metadata_columns

# Normalize any DataFrame with unknown datetime format
df = normalize_metadata_columns(df)
# Result: df has 'datetime' and 'ticker' columns, timezone-aware but not localized
```

### Pattern 1: Fix datetime in a pipeline stage
```python
async def run(self, **kwargs) -> Dict[str, Any]:
    data = kwargs.get('input_data')
    
    # At stage entry: normalize metadata
    data = normalize_metadata_columns(data)
    
    # ... process data ...
    
    # At stage exit: ensure properly formatted
    assert 'datetime' in data.columns
    assert 'ticker' in data.columns
    
    return {'output_data': data}
```

### Pattern 2: Handle multiple data sources
```python
cleaned_data = kwargs.get('cleaned_data')

# Normalize prices by timeframe
for tf, df in cleaned_data['prices'].items():
    cleaned_data['prices'][tf] = normalize_metadata_columns(df)

# Normalize news
cleaned_data['news'] = normalize_metadata_columns(cleaned_data['news'])

# Normalize macro
if 'macro_data' in cleaned_data:
    cleaned_data['macro_data'] = normalize_metadata_columns(cleaned_data['macro_data'])
```

### Pattern 3: Prepare features for model input
```python
from src.features.utils.datetime_utils import split_datetime_ticker

# Split features from metadata for model input
features_df, metadata_df = split_datetime_ticker(enriched_data)

# Train model on features only (no datetime/ticker to prevent data leakage)
X = features_df[feature_cols]
y = targets_df['target_column']
model.fit(X, y)

# Later, reconstruct full dataframe for predictions
predictions_with_metadata = roundtrip_datetime_ticker(predictions, metadata_df)
```

## Data Flow Quick Reference

### Stage Input/Output Summary
```
Stage 0: {} → {}
Stage 1: CLI args → {'raw_data': {source: DataFrame, ...}}
Stage 2: {'raw_data': ...} → {'cleaned_data': {'prices': {tf: df}, 'news': df, 'macro_data': df}}
Stage 3: {'cleaned_data': ...} → {'enriched_data': DataFrame}
Stage 4: {'enriched_data': ...} → {'models_metadata': {...}, 'processed_data': ...}
Stage 5: {'enriched_data': ..., 'models_metadata': ...} → {'prediction_results': {...}}
Stage 6: {'prediction_results': ...} → {'trading_signals': [...]}
Stage 7: {'trading_signals': ...} → {'evaluation_summary': {...}}
```

### Required Columns at Each Stage
```
Stage 2 Output: datetime, ticker (in all dataframes)
Stage 3 Output: datetime, ticker, target_* columns
Stage 4 Input:  datetime, ticker (in enriched_data)
Stage 5 Input:  datetime, ticker (in features_df and models_metadata)
```

## Common Issues & Solutions

### Issue: "datetime not found in DataFrame"
```python
# Solution 1: Use the utility
df = ensure_datetime_column(df, raise_on_missing=False)

# Solution 2: Handle specific case
if df.index.name == 'datetime':
    df = df.reset_index()
elif 'published_at' in df.columns:
    df['datetime'] = df['published_at']
```

### Issue: "Timezone comparison error in merge"
```python
# Solution: Normalize timezone
df['datetime'] = pd.to_datetime(df['datetime']).dt.tz_localize(None)

# Then safe to merge:
merged = df1.merge(df2, on=['datetime', 'ticker'])
```

### Issue: "selected_features file not found"
```python
# Solution: Use the multiple fallback patterns in Stage 4
file_candidates = [
    batch_dir / f"selected_features_{model}_{ticker}_{target}.json",
    batch_dir / f"selected_features_{model}_{ticker}.json",
    batch_dir / f"selected_features_{model}.json"
]

for candidate in file_candidates:
    if candidate.exists():
        with open(candidate, 'r') as f:
            features = json.load(f).get('selected_features', [])
```

### Issue: "models_metadata missing in Stage 5"
```python
# Solution: Fallback disk loading
if not models_metadata:
    models_metadata = stage_5_results.get('models_metadata', {})
    if not models_metadata:
        # Try loading from disk
        models_path = Path('trained_models/models_metadata.json')
        if models_path.exists():
            with open(models_path) as f:
                models_metadata = json.load(f)
```

## Testing Utilities

### Run the data flow test suite
```bash
# All tests
pytest tests/pipeline/test_data_flow.py -v

# Specific test
pytest tests/pipeline/test_data_flow.py::TestDataFlowValidation::test_datetime_column_normalization -v

# With output capture (see print statements)
pytest tests/pipeline/test_data_flow.py -v -s
```

### Quick validation script
```python
import pandas as pd
from src.features.utils.datetime_utils import normalize_metadata_columns

# Test with your data
df = pd.read_parquet("your_data.parquet")
df_normalized = normalize_metadata_columns(df)

print(f"Has datetime: {'datetime' in df_normalized.columns}")
print(f"Has ticker: {'ticker' in df_normalized.columns}")
print(f"datetime type: {df_normalized['datetime'].dtype}")
print(f"datetime timezone: {df_normalized['datetime'].dt.tz}")
```

## Execution Commands

### Local Mode Only
```bash
python run_hybrid_pipeline.py --mode local
```

### Fast Test (Single Ticker, Fewer Epochs)
```bash
python run_hybrid_pipeline.py --mode local --test-ticker AMD --epochs 5 --max-iterations 10
```

### Light Models Only
```bash
python run_hybrid_pipeline.py --mode light --test-ticker AMD
```

### Prepare for Colab
```bash
python run_hybrid_pipeline.py --mode prepare --batch-name my_batch
```

### Full Hybrid Pipeline
```bash
python run_hybrid_pipeline.py --mode full
```

### Continue from saved state
```bash
python run_hybrid_pipeline.py --mode continue --batch-name my_batch --stages 5 6 7
```

## Configuration Files

### Key Config Locations
```
src/config/paths.yaml          - Directory paths
src/config/assets.yaml         - Ticker presets
src/config/models.yaml         - Model definitions
src/config/targets.yaml        - Target definitions
src/config/features.yaml       - Feature enrichers
src/config/system.yaml         - System settings
unified_config.yaml            - Master config
```

### Runtime Parameters (Auto-generated)
```
src/config/runtime_params.json  - Runtime settings (created by run_hybrid_pipeline.py)
data/colab/accumulated/batch_name/  - Batch directory with data
```

## Debugging Tips

### Enable Verbose Logging
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Or in code:
from src.core.logging.logger import ProjectLogger
logger = ProjectLogger.get_logger(__name__)
logger.debug("Debug message")
```

### Check Data at Each Stage
```python
# Add this after each stage in pipeline.run():
import json
stage_name = type(stage).__name__
checkpoint_file = f"checkpoint_{i}_{stage_name}.json"

# For DataFrames:
if isinstance(stage_output, dict):
    for key, value in stage_output.items():
        if isinstance(value, pd.DataFrame):
            value.to_parquet(f"{checkpoint_file}_{key}.parquet")
            logger.info(f"Saved checkpoint: {key}")
```

### Profile Memory Usage
```python
import psutil
import os

process = psutil.Process(os.getpid())
mem_before = process.memory_info().rss / 1024 / 1024  # MB

# ... run stage ...

mem_after = process.memory_info().rss / 1024 / 1024
logger.info(f"Memory delta: {mem_after - mem_before:.1f} MB")
```

## Import Cheatsheet

```python
# Datetime utilities
from src.features.utils.datetime_utils import (
    ensure_datetime_column,
    ensure_ticker_column,
    normalize_metadata_columns,
    split_datetime_ticker,
    roundtrip_datetime_ticker,
    deduplicate_on_metadata,
    ensure_datetime_sorted
)

# Pipeline components
from src.pipeline.pipeline_orchestrator import PipelineOrchestrator
from src.pipeline.hybrid_orchestrator import HybridOrchestrator
from src.config.unified_config_manager import UnifiedConfigManager

# Logging
from src.core.logging.logger import ProjectLogger
logger = ProjectLogger.get_logger(__name__)

# Error handling
from src.core.error_handling.error_handler import ErrorHandler
```

## Support & Troubleshooting

### Documentation Files
- `PIPELINE_AUDIT_SESSION2_SUMMARY.md` - Complete audit results
- `SESSION2_AUDIT_COMPLETE.md` - Operational validation
- `src/features/utils/datetime_utils.py` - Source code with docstrings
- `tests/pipeline/test_data_flow.py` - Test examples

### Quick Contact Points
1. Check logs: `data/logs/` directory
2. Review test case: `tests/pipeline/test_data_flow.py`
3. Trace data: `checkpoint_*.parquet` files
4. Verify config: `src/config/runtime_params.json`

---

**Last Updated:** Session 2
**Version:** 1.0
