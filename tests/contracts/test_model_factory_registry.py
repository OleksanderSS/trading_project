
import importlib, sys
from pathlib import Path

def test_import_model_factory_does_not_import_tensorflow_or_transformers():
    sys.modules.pop('tensorflow',None); sys.modules.pop('transformers',None)
    importlib.import_module('src.factories.model_factory')
    assert 'tensorflow' not in sys.modules, 'Importing ModelFactory should not import TensorFlow'
    assert 'transformers' not in sys.modules, 'Importing ModelFactory should not import HuggingFace transformers'
def test_model_registry_not_unused_if_present():
    registry=Path('src/config/model_registry.py'); factory=Path('src/factories/model_factory.py')
    if not registry.exists() or not factory.exists(): return
    text=factory.read_text(encoding='utf-8',errors='ignore')
    assert 'model_registry' in text or 'ModelRegistry' in text, 'model_registry.py exists but ModelFactory does not appear to use it.'
