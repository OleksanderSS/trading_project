"""
Tests for PrototypeRegistry class
"""

import pytest
import json
import tempfile
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Union, cast
from src.models.prototypes.prototype import ModelPrototype
from src.models.prototypes.registry import PrototypeRegistry, get_prototype_registry
from src.models.interfaces import BaseModel


class MockModel(BaseModel):
    """Mock model for testing"""

    def __init__(self, **kwargs):
        super().__init__("mock", "regression")
        self.params = kwargs

    @property
    def name(self) -> str:
        """Override name property for testing"""
        return "mock_model"

    def train(
        self, 
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        **kwargs
    ) -> Dict[str, Any]:
        """Mock train implementation"""
        self.is_trained = True
        return {"status": "trained", "samples": len(X) if hasattr(X, '__len__') else 0}

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Mock predict implementation"""
        length = len(X) if hasattr(X, '__len__') else 1
        return np.array([0.5] * length)

    def load_model(self, path: str):
        """Mock load_model implementation"""
        pass

    def save_model(self, path: str):
        """Mock save_model implementation"""
        pass


class TestPrototypeRegistry:
    """Test suite for PrototypeRegistry"""

    @pytest.fixture
    def temp_registry_path(self):
        """Create temporary registry path"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield str(Path(tmpdir) / "registry.json")

    @pytest.fixture
    def registry(self, temp_registry_path):
        """Create registry with temporary path"""
        return PrototypeRegistry(registry_path=temp_registry_path)

    @pytest.fixture
    def sample_prototype(self):
        """Create sample prototype"""
        return ModelPrototype(
            model_id="test_model",
            model_class=cast(type[BaseModel], MockModel),
            version="1.0.0",
            dependencies=[],
            metadata={"param1": "value1"},
        )

    def test_registry_creation(self, registry):
        """Test registry creation"""
        assert registry is not None
        assert len(registry.prototypes) == 0

    def test_register_prototype(self, registry, sample_prototype):
        """Test prototype registration"""
        result = registry.register(sample_prototype)

        assert result is True
        assert "test_model" in registry.prototypes
        assert registry.prototypes["test_model"] == sample_prototype

    def test_register_multiple_prototypes(self, registry):
        """Test registering multiple prototypes"""
        proto1 = ModelPrototype(
            model_id="model1",
            model_class=cast(type[BaseModel], MockModel),
            version="1.0.0",
            dependencies=[],
            metadata={},
        )
        proto2 = ModelPrototype(
            model_id="model2",
            model_class=cast(type[BaseModel], MockModel),
            version="1.0.0",
            dependencies=[],
            metadata={},
        )

        registry.register(proto1)
        registry.register(proto2)

        assert len(registry.prototypes) == 2
        assert "model1" in registry.prototypes
        assert "model2" in registry.prototypes

    def test_get_prototype(self, registry, sample_prototype):
        """Test prototype retrieval"""
        registry.register(sample_prototype)
        retrieved = registry.get("test_model")

        assert retrieved is not None
        assert retrieved.model_id == "test_model"

    def test_get_nonexistent_prototype(self, registry):
        """Test retrieving nonexistent prototype"""
        retrieved = registry.get("nonexistent")
        assert retrieved is None

    def test_clone_from_registry(self, registry, sample_prototype):
        """Test cloning model from registry"""
        registry.register(sample_prototype)
        model = registry.clone("test_model", param1="new_value")

        assert model is not None
        assert isinstance(model, MockModel)
        assert model.params["param1"] == "new_value"

    def test_clone_nonexistent_prototype(self, registry):
        """Test cloning nonexistent prototype"""
        model = registry.clone("nonexistent")
        assert model is None

    def test_list_all_prototypes(self, registry):
        """Test listing all prototypes"""
        proto1 = ModelPrototype(
            model_id="model1",
            model_class=cast(type[BaseModel], MockModel),
            version="1.0.0",
            dependencies=[],
            metadata={},
        )
        proto2 = ModelPrototype(
            model_id="model2",
            model_class=cast(type[BaseModel], MockModel),
            version="1.0.0",
            dependencies=[],
            metadata={},
        )

        registry.register(proto1)
        registry.register(proto2)

        all_ids = registry.list_all()
        assert len(all_ids) == 2
        assert "model1" in all_ids
        assert "model2" in all_ids

    def test_get_by_type(self, registry):
        """Test filtering prototypes by type"""
        proto_catboost = ModelPrototype(
            model_id="catboost_v1",
            model_class=cast(type[BaseModel], MockModel),
            version="1.0.0",
            dependencies=[],
            metadata={},
        )
        proto_lstm = ModelPrototype(
            model_id="lstm_v1",
            model_class=cast(type[BaseModel], MockModel),
            version="1.0.0",
            dependencies=[],
            metadata={},
        )

        registry.register(proto_catboost)
        registry.register(proto_lstm)

        catboost_models = registry.get_by_type("catboost")
        assert len(catboost_models) == 1
        assert catboost_models[0].model_id == "catboost_v1"

    def test_get_by_version(self, registry):
        """Test retrieving specific version"""
        proto = ModelPrototype(
            model_id="catboost_v1",
            model_class=cast(type[BaseModel], MockModel),
            version="1.0.0",
            dependencies=[],
            metadata={},
        )

        registry.register(proto)
        retrieved = registry.get_by_version("catboost", "1")

        assert retrieved is not None
        assert retrieved.model_id == "catboost_v1"

    def test_remove_prototype(self, registry, sample_prototype):
        """Test prototype removal"""
        registry.register(sample_prototype)
        assert "test_model" in registry.prototypes

        result = registry.remove("test_model")
        assert result is True
        assert "test_model" not in registry.prototypes

    def test_remove_nonexistent_prototype(self, registry):
        """Test removing nonexistent prototype"""
        result = registry.remove("nonexistent")
        assert result is False

    def test_registry_persistence(self, temp_registry_path):
        """Test registry persistence to disk"""
        # Create and populate registry
        registry1 = PrototypeRegistry(registry_path=temp_registry_path)
        proto = ModelPrototype(
            model_id="test_model",
            model_class=cast(type[BaseModel], MockModel),
            version="1.0.0",
            dependencies=[],
            metadata={"param1": "value1"},
        )
        registry1.register(proto)

        # Load registry from disk
        registry2 = PrototypeRegistry(registry_path=temp_registry_path)
        assert len(registry2.prototypes) == 1
        assert "test_model" in registry2.prototypes

    def test_export_summary(self, registry, sample_prototype):
        """Test registry summary export"""
        registry.register(sample_prototype)
        summary = registry.export_summary()

        assert "total_prototypes" in summary
        assert "prototypes" in summary
        assert "registry_path" in summary
        assert summary["total_prototypes"] == 1

    def test_get_stats(self, registry):
        """Test registry statistics"""
        proto1 = ModelPrototype(
            model_id="model1",
            model_class=cast(type[BaseModel], MockModel),
            version="1.0.0",
            dependencies=[],
            metadata={},
        )
        proto2 = ModelPrototype(
            model_id="model2",
            model_class=cast(type[BaseModel], MockModel),
            version="1.0.0",
            dependencies=[],
            metadata={},
        )

        registry.register(proto1)
        registry.register(proto2)

        # Clone some models
        registry.clone("model1")
        registry.clone("model1")
        registry.clone("model2")

        stats = registry.get_stats()

        assert stats["total_prototypes"] == 2
        assert stats["total_clones"] == 3
        assert stats["avg_clones_per_prototype"] == 1.5

    def test_registry_repr(self, registry, sample_prototype):
        """Test registry string representation"""
        registry.register(sample_prototype)
        repr_str = repr(registry)

        assert "PrototypeRegistry" in repr_str
        assert "1" in repr_str  # Number of prototypes

    def test_register_overwrites_existing(self, registry):
        """Test that registering overwrites existing prototype"""
        proto1 = ModelPrototype(
            model_id="test_model",
            model_class=cast(type[BaseModel], MockModel),
            version="1.0.0",
            dependencies=[],
            metadata={"param1": "value1"},
        )
        proto2 = ModelPrototype(
            model_id="test_model",
            model_class=cast(type[BaseModel], MockModel),
            version="2.0.0",
            dependencies=[],
            metadata={"param1": "value2"},
        )

        registry.register(proto1)
        registry.register(proto2)

        assert len(registry.prototypes) == 1
        assert registry.prototypes["test_model"].version == "2.0.0"

    def test_get_by_type_case_insensitive(self, registry):
        """Test type filtering is case-insensitive"""
        proto = ModelPrototype(
            model_id="CatBoost_v1",
            model_class=cast(type[BaseModel], MockModel),
            version="1.0.0",
            dependencies=[],
            metadata={},
        )

        registry.register(proto)

        # Should find with lowercase
        results = registry.get_by_type("catboost")
        assert len(results) == 1

        # Should find with uppercase
        results = registry.get_by_type("CATBOOST")
        assert len(results) == 1


class TestPrototypeRegistrySingleton:
    """Test suite for singleton pattern"""

    def test_get_prototype_registry_singleton(self):
        """Test that get_prototype_registry returns singleton"""
        registry1 = get_prototype_registry()
        registry2 = get_prototype_registry()

        assert registry1 is registry2

    def test_singleton_persistence(self):
        """Test that singleton persists across calls"""
        registry = get_prototype_registry()

        proto = ModelPrototype(
            model_id="singleton_test",
            model_class=cast(type[BaseModel], MockModel),
            version="1.0.0",
            dependencies=[],
            metadata={},
        )

        registry.register(proto)

        # Get singleton again
        registry2 = get_prototype_registry()
        assert "singleton_test" in registry2.prototypes
