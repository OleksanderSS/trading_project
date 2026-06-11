#!/usr/bin/env python3
"""
Intelligent Data Filter - Facade for Modular Data Filtering System.
Maintains backward compatibility with the original IntelligentDataFilter.
"""

from .filters.orchestrator import IntelligentDataFilter as ModularIntelligentDataFilter


class IntelligentDataFilter(ModularIntelligentDataFilter):
    """
    Facade for IntelligentDataFilter.
    Delegates to the modular components in the 'filters' subdirectory.
    """
    pass

def filter_data_for_model_training(raw_data: dict, config: dict | None = None) -> dict:
    """
    Convenience function for data filtering.
    """
    filter_obj = IntelligentDataFilter(config)
    return filter_obj.filter_quality_data(raw_data)
