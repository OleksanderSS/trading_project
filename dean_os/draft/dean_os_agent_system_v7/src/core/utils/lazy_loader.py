import importlib
from typing import Any


class LazyLoader:
    """
    A utility class to lazily import heavy modules only when they are accessed.
    """
    def __init__(self, module_name: str, package: str = None):
        self.module_name = module_name
        self.package = package
        self._module = None

    def __getattr__(self, name: str) -> Any:
        if self._module is None:
            self._module = importlib.import_module(self.module_name, package=self.package)
        return getattr(self._module, name)

# Standard instances for heavy libraries
tf = LazyLoader("tensorflow")
torch = LazyLoader("torch")
xgb = LazyLoader("xgboost")
lgb = LazyLoader("lightgbm")
catboost = LazyLoader("catboost")
