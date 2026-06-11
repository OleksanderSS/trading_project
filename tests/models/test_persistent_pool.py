"""Tests for PersistentModelPool."""

import pytest
import tempfile
import shutil
from pathlib import Path

from src.models.persistent_pool import PersistentModelPool


class MockModel:
    """Mock model for testing."""
    def __init__(self, name="mock", **kwargs):
        self.name = name
        self.params = kwargs
    
    def predict(self, X):
        return [0.5] * len(X)


@pytest.fixture
def temp_cache_dir():
    """Create temporary cache directory."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


def test_persistent_pool_creation(temp_cache_dir):
    """Test persistent pool creation."""
    pool = PersistentModelPool(max_models=5, cache_dir=temp_cache_dir)
    
    assert pool.max_models == 5
    assert pool.cache_dir == Path(temp_cache_dir)
    assert pool.cache_dir.exists()
    assert len(pool.model_metadata) == 0
    assert len(pool.quality_scores) == 0


def test_add_model_with_metadata(temp_cache_dir):
    """Test adding model with metadata."""
    pool = PersistentModelPool(cache_dir=temp_cache_dir)
    model = MockModel("test_model")
    
    pool.add_model_with_metadata(
        "test_model",
        model,
        metadata={"version": "1.0", "ticker": "BTC"},
        quality_score=0.85
    )
    
    assert "test_model" in pool.models
    assert "test_model" in pool.model_metadata
    assert "test_model" in pool.quality_scores
    assert pool.quality_scores["test_model"] == 0.85
    assert pool.model_metadata["test_model"]["version"] == "1.0"
    assert pool.model_metadata["test_model"]["ticker"] == "BTC"


def test_cache_index_persistence(temp_cache_dir):
    """Test cache index persists across instances."""
    # Create pool and add model
    pool1 = PersistentModelPool(cache_dir=temp_cache_dir)
    model = MockModel("test_model")
    pool1.add_model_with_metadata(
        "test_model", model,
        metadata={"version": "1.0"},
        quality_score=0.85
    )
    
    # Create new pool instance
    pool2 = PersistentModelPool(cache_dir=temp_cache_dir)
    
    # Check metadata persisted
    assert "test_model" in pool2.model_metadata
    assert "test_model" in pool2.quality_scores
    assert pool2.quality_scores["test_model"] == 0.85


def test_get_model_with_quality_check(temp_cache_dir):
    """Test getting model with quality check."""
    pool = PersistentModelPool(cache_dir=temp_cache_dir)
    model = MockModel("high_quality")
    
    pool.add_model_with_metadata(
        "high_quality", model,
        metadata={},
        quality_score=0.85
    )
    
    # Should pass quality check
    retrieved = pool.get_model_with_quality_check(
        "high_quality",
        lambda: model,
        min_quality=0.7
    )
    assert retrieved is not None
    
    # Add low quality model
    low_model = MockModel("low_quality")
    pool.add_model_with_metadata(
        "low_quality", low_model,
        metadata={},
        quality_score=0.3
    )
    
    # Should fail quality check
    retrieved = pool.get_model_with_quality_check(
        "low_quality",
        lambda: low_model,
        min_quality=0.7
    )
    assert retrieved is None


def test_warm_up(temp_cache_dir):
    """Test warm-up mechanism."""
    pool = PersistentModelPool(max_models=10, cache_dir=temp_cache_dir)
    
    # Create loader functions
    loader_fns = {
        "model1": lambda: MockModel("model1"),
        "model2": lambda: MockModel("model2"),
        "model3": lambda: MockModel("model3")
    }
    
    # Warm up
    pool.warm_up(["model1", "model2", "model3"], loader_fns)
    
    # Check all models loaded
    assert len(pool.models) == 3
    assert "model1" in pool.models
    assert "model2" in pool.models
    assert "model3" in pool.models


def test_update_quality_score(temp_cache_dir):
    """Test updating quality score."""
    pool = PersistentModelPool(cache_dir=temp_cache_dir)
    model = MockModel("test_model")
    
    pool.add_model_with_metadata(
        "test_model", model,
        metadata={},
        quality_score=0.5
    )
    
    # Update quality
    pool.update_quality_score("test_model", 0.9)
    
    assert pool.quality_scores["test_model"] == 0.9


def test_get_metadata(temp_cache_dir):
    """Test getting metadata."""
    pool = PersistentModelPool(cache_dir=temp_cache_dir)
    model = MockModel("test_model")
    
    metadata = {"version": "2.0", "ticker": "ETH"}
    pool.add_model_with_metadata(
        "test_model", model,
        metadata=metadata,
        quality_score=0.8
    )
    
    retrieved_metadata = pool.get_metadata("test_model")
    assert retrieved_metadata is not None
    assert retrieved_metadata["version"] == "2.0"
    assert retrieved_metadata["ticker"] == "ETH"


def test_get_enhanced_stats(temp_cache_dir):
    """Test enhanced statistics."""
    pool = PersistentModelPool(cache_dir=temp_cache_dir)
    
    # Add models with different quality scores
    for i in range(3):
        model = MockModel(f"model{i}")
        pool.add_model_with_metadata(
            f"model{i}", model,
            metadata={},
            quality_score=0.5 + i * 0.2
        )
    
    stats = pool.get_enhanced_stats()
    
    assert 'avg_quality' in stats
    assert 'models_with_metadata' in stats
    assert 'cache_dir' in stats
    assert stats['models_with_metadata'] == 3
    assert 0.5 < stats['avg_quality'] < 1.0


def test_lru_eviction_with_metadata(temp_cache_dir):
    """Test LRU eviction preserves metadata."""
    pool = PersistentModelPool(max_models=2, cache_dir=temp_cache_dir)
    
    # Add 3 models (should evict first)
    for i in range(3):
        model = MockModel(f"model{i}")
        pool.add_model_with_metadata(
            f"model{i}", model,
            metadata={"index": i},
            quality_score=0.5 + i * 0.1
        )
    
    # model0 should be evicted from pool
    assert "model0" not in pool.models
    assert "model1" in pool.models
    assert "model2" in pool.models
    
    # But metadata should still exist
    assert "model0" in pool.model_metadata
    assert "model0" in pool.quality_scores


def test_export_to_disk(temp_cache_dir):
    """Test exporting models to disk."""
    pool = PersistentModelPool(cache_dir=temp_cache_dir)
    
    # Add models
    for i in range(3):
        model = MockModel(f"model{i}")
        pool.add_model_with_metadata(
            f"model{i}", model,
            metadata={},
            quality_score=0.8
        )
    
    # Export
    export_dir = Path(temp_cache_dir) / "exports"
    pool.export_to_disk(str(export_dir))
    
    # Check files created
    assert export_dir.exists()
    assert (export_dir / "model0.joblib").exists()
    assert (export_dir / "model1.joblib").exists()
    assert (export_dir / "model2.joblib").exists()
