"""
Data Flow Validation Test for Trading Pipeline Stages 0-5

This test validates that data passes correctly between pipeline stages,
ensuring proper column naming, data types, and structure consistency.

Test Coverage:
- Stage 0: Environment setup
- Stage 1: Data collection
- Stage 2: Data processing (datetime/ticker normalization)
- Stage 3: Feature engineering (enrichment)
- Stage 4: Modeling (champion models)
- Stage 5: Prediction generation

Run with: python -m pytest tests/pipeline/test_data_flow.py -v
"""

import asyncio
import json
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any

from src.config.unified_config_manager import UnifiedConfigManager
from src.core.error_handling.error_handler import ErrorHandler
from src.data.management.data_manager import DataManager
from src.features.utils.datetime_utils import ensure_datetime_column, normalize_metadata_columns


# Fixtures
@pytest.fixture
def config_manager():
    """Loads the unified configuration."""
    return UnifiedConfigManager()


@pytest.fixture
def error_handler(config_manager):
    """Creates an error handler."""
    return ErrorHandler(config_manager)


@pytest.fixture
def db_manager(config_manager):
    """Creates a data manager."""
    return DataManager(config_manager)


# Test Cases
class TestDataFlowValidation:
    """Validates data flow through pipeline stages."""

    def test_datetime_column_normalization(self):
        """Test that datetime_utils properly normalizes datetime columns."""
        # Case 1: datetime as column
        df = pd.DataFrame({
            'datetime': pd.date_range('2024-01-01', periods=5),
            'value': range(5)
        })
        result = normalize_metadata_columns(df)
        assert 'datetime' in result.columns
        assert not result['datetime'].dt.tz is None or result['datetime'].dt.tz.zone == 'UTC'
        
        # Case 2: datetime as index
        df = pd.DataFrame({
            'value': range(5)
        }, index=pd.date_range('2024-01-01', periods=5, name='datetime'))
        result = normalize_metadata_columns(df)
        assert 'datetime' in result.columns
        assert isinstance(result['datetime'].iloc[0], pd.Timestamp)

    def test_ticker_column_presence(self):
        """Test that ticker column is properly added."""
        df = pd.DataFrame({
            'value': range(5)
        })
        result = normalize_metadata_columns(df)
        assert 'ticker' in result.columns

    def test_stage_2_output_structure(self):
        """Test that Stage 2 returns properly structured cleaned_data."""
        # Simulate Stage 2 output
        stage2_output = {
            'cleaned_data': {
                'prices': {
                    '1d': pd.DataFrame({
                        'datetime': pd.date_range('2024-01-01', periods=5),
                        'ticker': ['AMD'] * 5,
                        'close': [100, 101, 102, 103, 104]
                    }),
                    '1h': pd.DataFrame({
                        'datetime': pd.date_range('2024-01-01', periods= 10, freq='H'),
                        'ticker': ['AMD'] * 10,
                        'close': np.random.randn(10).cumsum() + 100
                    })
                },
                'news': pd.DataFrame({
                    'datetime': pd.date_range('2024-01-01', periods=3),
                    'ticker': ['AMD'] * 3,
                    'title': ['News 1', 'News 2', 'News 3']
                })
            }
        }
        
        # Validate structure
        cleaned_data = stage2_output['cleaned_data']
        
        # Check prices structure
        assert 'prices' in cleaned_data
        assert isinstance(cleaned_data['prices'], dict)
        for tf, df in cleaned_data['prices'].items():
            assert isinstance(df, pd.DataFrame)
            assert 'datetime' in df.columns
            assert 'ticker' in df.columns
        
        # Check news structure
        assert 'news' in cleaned_data
        assert 'datetime' in cleaned_data['news'].columns
        assert 'ticker' in cleaned_data['news'].columns

    def test_stage_3_output_structure(self):
        """Test that Stage 3 returns properly structured enriched_data."""
        # Simulate Stage 3 output
        stage3_output = {
            'enriched_data': pd.DataFrame({
                'datetime': pd.date_range('2024-01-01', periods=5),
                'ticker': ['AMD'] * 5,
                'close': [100, 101, 102, 103, 104],
                'sma_20': [99, 100, 101, 102, 103],
                'rsi': [45, 50, 55, 60, 65],
                'target_return_1d': [0.01, 0.01, -0.01, 0.01, 0.00],
                'target_signal_1d': [1, 1, 0, 1, 0]
            }),
            'feature_version': '20240101_1200'
        }
        
        enriched_data = stage3_output['enriched_data']
        
        # Validate required columns
        assert 'datetime' in enriched_data.columns
        assert 'ticker' in enriched_data.columns
        
        # Validate target columns
        target_cols = [c for c in enriched_data.columns if c.startswith('target_')]
        assert len(target_cols) > 0, "No target columns found"
        
        # Validate feature columns (everything except metadata and targets)
        feature_cols = [c for c in enriched_data.columns 
                       if c not in ['datetime', 'ticker'] and not c.startswith('target_')]
        assert len(feature_cols) > 0, "No feature columns found"

    def test_stage_4_models_metadata_structure(self):
        """Test that Stage 4 returns properly structured models_metadata."""
        # Simulate Stage 4 output
        stage4_output = {
            'models_metadata': {
                'AMD_target_return_1d': {
                    'ticker': 'AMD',
                    'target': 'target_return_1d',
                    'winner': 'catboost',
                    'model_path': 'data/colab/accumulated/main_database/models/catboost_AMD_target_return_1d.pt',
                    'selected_features': ['sma_20', 'rsi', 'volatility', 'momentum'],
                    'metrics': {
                        'accuracy': 0.65,
                        'precision': 0.68,
                        'recall': 0.62
                    },
                    'context_fingerprint': 'ctx_hash_123',
                    'market_regime': 'neutral'
                }
            },
            'processed_data': pd.DataFrame()  # enriched_data from Stage 3
        }
        
        models_metadata = stage4_output['models_metadata']
        
        # Validate structure
        assert isinstance(models_metadata, dict)
        
        for context_id, meta in models_metadata.items():
            # Validate required keys
            assert 'ticker' in meta
            assert 'target' in meta
            assert 'winner' in meta
            assert 'model_path' in meta
            assert 'selected_features' in meta
            assert 'metrics' in meta
            assert 'context_fingerprint' in meta
            
            # Validate data types
            assert isinstance(meta['selected_features'], list)
            assert isinstance(meta['metrics'], dict)
            assert len(meta['selected_features']) > 0

    def test_stage_5_requires_models_metadata(self):
        """Test that Stage 5 properly handles models_metadata passing."""
        # Simulate Stage 5 input
        features_df = pd.DataFrame({
            'datetime': pd.date_range('2024-01-01', periods=5),
            'ticker': ['AMD'] * 5,
            'sma_20': [99, 100, 101, 102, 103],
            'rsi': [45, 50, 55, 60, 65]
        })
        
        models_metadata = {
            'AMD_target_return_1d': {
                'ticker': 'AMD',
                'target': 'target_return_1d',
                'winner': 'catboost',
                'model_path': 'data/colab/accumulated/main_database/models/catboost_AMD_target_return_1d.pt',
                'selected_features': ['sma_20', 'rsi']
            }
        }
        
        # Validate that features_df has required columns for model input
        for context_id, meta in models_metadata.items():
            selected_features = meta.get('selected_features', [])
            
            # Check that selected features exist in features_df
            available_features = [f for f in selected_features if f in features_df.columns]
            assert len(available_features) > 0, \
                f"No selected features available in features_df for {context_id}"
            
            # Metadata columns should always be present
            assert 'datetime' in features_df.columns
            assert 'ticker' in features_df.columns

    def test_end_to_end_data_flow(self):
        """Integration test for complete data flow through all stages."""
        # Create minimal test data mimicking real pipeline flow
        
        # Stage 1 output
        stage1 = {
            'raw_data': {
                'yahoo_finance': pd.DataFrame({
                    'datetime': pd.date_range('2024-01-01', periods=10),
                    'ticker': ['AMD'] * 10,
                    'open': np.random.randn(10).cumsum() + 100,
                    'high': np.random.randn(10).cumsum() + 101,
                    'low': np.random.randn(10).cumsum() + 99,
                    'close': np.random.randn(10).cumsum() + 100,
                    'volume': np.random.randint(1000000, 5000000, 10)
                }),
                'news': pd.DataFrame({
                    'datetime': pd.date_range('2024-01-01', periods=5),
                    'ticker': ['AMD'] * 5,
                    'title': [f'News {i}' for i in range(5)],
                    'sentiment': np.random.uniform(-1, 1, 5)
                })
            }
        }
        
        # Stage 2: Normalize
        cleaned_prices = normalize_metadata_columns(stage1['raw_data']['yahoo_finance'])
        cleaned_news = normalize_metadata_columns(stage1['raw_data']['news'])
        
        stage2_output = {
            'cleaned_data': {
                'prices': {'1d': cleaned_prices},
                'news': cleaned_news
            }
        }
        
        # Validate Stage 2 → 3 transition
        assert 'prices' in stage2_output['cleaned_data']
        assert isinstance(stage2_output['cleaned_data']['prices'], dict)
        
        for tf, df in stage2_output['cleaned_data']['prices'].items():
            assert all(col in df.columns for col in ['datetime', 'ticker'])
        
        # Stage 3: Create enriched data with targets
        enriched = cleaned_prices.copy()
        enriched['sma_20'] = enriched['close'].rolling(20).mean()
        enriched['target_return_1d'] = enriched['close'].pct_change().shift(-1)
        
        stage3_output = {
            'enriched_data': enriched
        }
        
        # Validate Stage 3 output
        assert 'datetime' in stage3_output['enriched_data'].columns
        assert 'ticker' in stage3_output['enriched_data'].columns
        target_cols = [c for c in stage3_output['enriched_data'].columns if c.startswith('target_')]
        assert len(target_cols) > 0
        
        # Stage 4: Models metadata
        stage4_output = {
            'models_metadata': {
                'AMD_target_return_1d': {
                    'ticker': 'AMD',
                    'target': 'target_return_1d',
                    'winner': 'catboost',
                    'model_path': 'models/catboost_AMD_target_return_1d.pt',
                    'selected_features': ['close', 'sma_20']
                }
            }
        }
        
        # Validate Stage 4 → 5 transition
        models_meta = stage4_output['models_metadata']
        for context_id, meta in models_meta.items():
            for feat in meta['selected_features']:
                assert feat in stage3_output['enriched_data'].columns, \
                    f"Feature {feat} not in enriched_data"


# Run tests
if __name__ == '__main__':
    pytest.main([__file__, '-v'])
