import pytest
from src.factories.tree_model_factory import TreeModelFactory
from src.models.tree.xgboost_model import XGBoostModel

def test_tree_model_factory_creation():
    # Перевіряємо створення моделі без конфігурації
    model = TreeModelFactory.create_model('XGBoost')
    assert isinstance(model, XGBoostModel)

def test_tree_model_factory_invalid_model():
    with pytest.raises(ValueError, match="not supported"):
        TreeModelFactory.create_model('InvalidModel')

def test_is_tree_model():
    assert TreeModelFactory.is_tree_model('XGBoost')
    assert not TreeModelFactory.is_tree_model('Linear')


def test_model_factory_normalizes_config_names():
    from src.factories.model_factory import ModelFactory

    assert ModelFactory._validate_and_normalize_name('transformer') == 'Transformer'
    assert ModelFactory._validate_and_normalize_name('lightgbm') == 'LightGBM'
    assert ModelFactory._validate_and_normalize_name('random_forest') == 'RandomForest'
    assert 'Transformer' in ModelFactory.get_available_models()
