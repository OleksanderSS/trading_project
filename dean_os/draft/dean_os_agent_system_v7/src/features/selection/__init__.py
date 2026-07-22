"""
Feature selection module - smart feature selection algorithms
"""

# Lazy imports to avoid heavy dependencies (DuckDB) on package import
# This allows import src.features.selection without DuckDB installed

__all__ = ["SmartFeatureSelector", "EnhancedSmartFeatureSelector", "get_enhanced_smart_selector", "select_features_enhanced"]


def __getattr__(name: str):
    """Lazy import for feature selection components."""
    if name == "SmartFeatureSelector":
        from .smart_selector import SmartFeatureSelector
        return SmartFeatureSelector
    elif name == "EnhancedSmartFeatureSelector" or name == "get_enhanced_smart_selector" or name == "select_features_enhanced":
        from .enhanced_smart_selector import (
            EnhancedSmartFeatureSelector,
            get_enhanced_smart_selector,
            select_features_enhanced,
        )
        if name == "EnhancedSmartFeatureSelector":
            return EnhancedSmartFeatureSelector
        elif name == "get_enhanced_smart_selector":
            return get_enhanced_smart_selector
        elif name == "select_features_enhanced":
            return select_features_enhanced

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
