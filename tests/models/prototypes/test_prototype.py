"""
Tests for ModelPrototype class
"""

import pytest
from datetime import datetime
import numpy as np
import pandas as pd
from typing import Dict, Any, Union, cast
from src.models.prototypes.prototype import ModelPrototype
from src.models.interfaces import BaseModel


class MockModel(BaseModel):
    """Mock model for testing"""

    def __init__(self, **kwargs):
        super().__init__(model_type="mock", task_type="testing")
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


class TestModelPrototype:
    """Test suite for ModelPrototype"""

    def _create_test_prototype(self, model_id="test_model", metadata=None, dependencies=None):
        """Helper method to create a test prototype with common parameters"""
        if metadata is None:
            metadata = {"param1": "value1"}
        if dependencies is None:
            dependencies = []
            
        return ModelPrototype(
            model_id=model_id,
            model_class=cast(type[BaseModel], MockModel),
            version="1.0.0",
            dependencies=dependencies,
            metadata=metadata,
        )

    def test_prototype_creation(self):
        """Test prototype creation with metadata"""
        prototype = self._create_test_prototype(metadata={"param1": "value1", "param2": 42})

        assert prototype.model_id == "test_model"
        assert prototype.version == "1.0.0"
        assert prototype.metadata["param1"] == "value1"
        assert prototype.metadata["param2"] == 42
        assert prototype._clone_count == 0

    def test_prototype_clone_basic(self):
        """Test basic prototype cloning"""
        prototype = self._create_test_prototype()

        model = prototype.clone()
        assert model is not None
        assert isinstance(model, MockModel)
        assert model.params["param1"] == "value1"
        assert prototype._clone_count == 1

    def test_prototype_clone_with_overrides(self):
        """Test prototype cloning with parameter overrides"""
        prototype = ModelPrototype(
            model_id="test_model",
            model_class=cast(type[BaseModel], MockModel),
            version="1.0.0",
            dependencies=[],
            metadata={"param1": "value1", "param2": 42},
        )

        model1 = prototype.clone(param1="new_value")
        model2 = prototype.clone(param2=100)

        # Type guards for BaseModel | None
        assert model1 is not None
        assert model2 is not None
        assert isinstance(model1, MockModel)
        assert isinstance(model2, MockModel)
        
        assert model1.params["param1"] == "new_value"
        assert model1.params["param2"] == 42  # From metadata
        assert model2.params["param1"] == "value1"  # From metadata
        assert model2.params["param2"] == 100
        assert prototype._clone_count == 2

    def test_prototype_clone_multiple(self):
        """Test multiple clones from same prototype"""
        prototype = self._create_test_prototype()

        models = [prototype.clone() for _ in range(5)]

        assert len(models) == 5
        assert all(isinstance(m, MockModel) for m in models)
        assert prototype._clone_count == 5

    def test_prototype_dependency_validation_success(self):
        """Test successful dependency validation"""
        prototype = self._create_test_prototype(
            dependencies=["json", "pathlib"],  # Built-in modules
            metadata={}
        )

        assert prototype.validate_dependencies() is True

    def test_prototype_dependency_validation_failure(self):
        """Test failed dependency validation"""
        prototype = self._create_test_prototype(
            dependencies=["nonexistent_package_xyz"],
            metadata={}
        )

        assert prototype.validate_dependencies() is False

    def test_prototype_clone_with_missing_dependencies(self):
        """Test clone fails with missing dependencies"""
        prototype = self._create_test_prototype(
            dependencies=["nonexistent_package_xyz"],
            metadata={}
        )

        model = prototype.clone()
        assert model is None

    def test_prototype_get_info(self):
        """Test prototype info retrieval"""
        prototype = self._create_test_prototype(
            dependencies=["json"],
            metadata={"param1": "value1"}
        )

        # Clone once to set validated flag
        prototype.clone()

        info = prototype.get_info()

        assert info["model_id"] == "test_model"
        assert info["model_class"] == "MockModel"
        assert info["version"] == "1.0.0"
        assert info["dependencies"] == ["json"]
        assert info["metadata"]["param1"] == "value1"
        assert info["validated"] is True
        assert info["clone_count"] == 1
        assert "created_at" in info

    def test_prototype_repr(self):
        """Test prototype string representation"""
        prototype = self._create_test_prototype(metadata={})

        repr_str = repr(prototype)
        assert "test_model" in repr_str
        assert "1.0.0" in repr_str

    def test_prototype_str(self):
        """Test prototype string conversion"""
        prototype = self._create_test_prototype(metadata={})

        str_repr = str(prototype)
        assert str_repr == "test_model v1.0.0"

    def test_prototype_created_at(self):
        """Test prototype creation timestamp"""
        before = datetime.now()
        prototype = self._create_test_prototype(metadata={})
        after = datetime.now()

        assert before <= prototype.created_at <= after

    def test_prototype_empty_metadata(self):
        """Test prototype with empty metadata"""
        prototype = self._create_test_prototype(metadata={})

        assert prototype.metadata == {}
        model = prototype.clone()
        assert model is not None

    def test_prototype_empty_dependencies(self):
        """Test prototype with empty dependencies"""
        prototype = self._create_test_prototype()

        assert prototype.dependencies == []
        assert prototype.validate_dependencies() is True
        model = prototype.clone()
        assert model is not None
