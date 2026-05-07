"""
Unit tests for DataManager security fixes
"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock
from src.data.management.data_manager import DataManager
from src.config.unified_config_manager import UnifiedConfigManager


class TestDataManagerSecurity:
    """Test security improvements in DataManager"""
    
    @pytest.fixture
    def data_manager(self):
        """Create DataManager instance for testing"""
        config_manager = Mock(spec=UnifiedConfigManager)
        config_manager.get.return_value = ':memory:'
        return DataManager(config_manager)
    
    def test_upsert_creates_table(self, data_manager):
        """Test that upsert creates table if not exists"""
        df = pd.DataFrame({
            'ticker': ['AMD', 'NVDA'],
            'timestamp': ['2026-01-01', '2026-01-02'],
            'close': [100.0, 200.0]
        })
        
        data_manager.upsert('test_table', df, unique_on=['ticker', 'timestamp'])
        
        assert data_manager.table_exists('test_table')
        result = data_manager.fetch_data_from_table('test_table')
        assert len(result) == 2
    
    def test_upsert_prevents_duplicates(self, data_manager):
        """Test that upsert prevents duplicate records atomically"""
        df1 = pd.DataFrame({
            'ticker': ['AMD'],
            'timestamp': ['2026-01-01'],
            'close': [100.0],
            'hash': ['hash1']  # Add hash for deduplication
        })
        
        df2 = pd.DataFrame({
            'ticker': ['AMD'],
            'timestamp': ['2026-01-01'],
            'close': [101.0],  # Different value, same key
            'hash': ['hash1']  # Same hash - should be deduplicated
        })
        
        # Insert first batch
        data_manager.upsert('test_table', df1)
        
        # Try to insert duplicate
        data_manager.upsert('test_table', df2)
        
        result = data_manager.fetch_data_from_table('test_table')
        assert len(result) == 1  # Should only have one record
        assert result['close'].iloc[0] == 100.0  # First value preserved
    
    def test_clean_numeric_data_removes_inf(self, data_manager):
        """Test that _clean_numeric_data removes Inf values"""
        df = pd.DataFrame({
            'value': [1.0, np.inf, 3.0, -np.inf, 5.0]
        })
        
        cleaned = data_manager._clean_numeric_data(df, 'test_table')
        
        assert not np.isinf(cleaned['value']).any()
    
    def test_clean_numeric_data_handles_nan(self, data_manager):
        """Test that _clean_numeric_data handles NaN values"""
        df = pd.DataFrame({
            'value': [1.0, np.nan, 3.0, np.nan, 5.0]
        })
        
        cleaned = data_manager._clean_numeric_data(df, 'test_table')
        
        # Should have no NaN after cleaning
        assert not cleaned['value'].isna().any()
    
    def test_should_checkpoint_critical_tables(self, data_manager):
        """Test that critical tables trigger checkpoints"""
        assert data_manager._should_checkpoint('enriched_features')
        assert data_manager._should_checkpoint('targets')
        assert data_manager._should_checkpoint('model_results')
        assert not data_manager._should_checkpoint('temp_table')
    
    def test_upsert_with_hash_column(self, data_manager):
        """Test upsert with hash-based deduplication"""
        df1 = pd.DataFrame({
            'hash': ['abc123', 'def456'],
            'data': ['value1', 'value2']
        })
        
        df2 = pd.DataFrame({
            'hash': ['abc123', 'ghi789'],  # One duplicate, one new
            'data': ['value1_new', 'value3']
        })
        
        data_manager.upsert('hash_table', df1)
        data_manager.upsert('hash_table', df2)
        
        result = data_manager.fetch_data_from_table('hash_table')
        assert len(result) == 3  # Should have 3 unique hashes
        
        # Verify original value preserved
        original = result[result['hash'] == 'abc123']
        assert original['data'].iloc[0] == 'value1'


class TestPathTraversalFix:
    """Test path traversal security fix"""
    
    def test_sanitize_path_input_imported(self):
        """Test that sanitize_path_input can be imported"""
        from src.training.progressive_trainer import sanitize_path_input
        assert callable(sanitize_path_input)
    
    def test_sanitize_path_blocks_traversal(self):
        """Test that path traversal is blocked"""
        from src.training.progressive_trainer import sanitize_path_input
        
        with pytest.raises(ValueError, match="Path traversal"):
            sanitize_path_input("../etc/passwd")
    
    def test_sanitize_path_blocks_absolute(self):
        """Test that absolute paths are blocked"""
        from src.training.progressive_trainer import sanitize_path_input
        
        with pytest.raises(ValueError, match="Absolute paths"):
            sanitize_path_input("/etc/passwd")
    
    def test_sanitize_path_blocks_null_byte(self):
        """Test that null bytes are blocked"""
        from src.training.progressive_trainer import sanitize_path_input
        
        with pytest.raises(ValueError, match="Null byte"):
            sanitize_path_input("file\x00.txt")
    
    def test_sanitize_path_allows_valid(self):
        """Test that valid paths are allowed"""
        from src.training.progressive_trainer import sanitize_path_input
        
        result = sanitize_path_input("checkpoint_batch_5.json")
        assert "checkpoint_batch_5.json" in result


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
