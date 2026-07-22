"""
Legacy compatibility module for ExperienceDiary.

This module provides a compatibility alias for the renamed ExperienceDiaryEngine class.
The ExperienceDiary class has been renamed to ExperienceDiaryEngine for clarity.

DEPRECATED: Use ExperienceDiaryEngine from src.meta_learning.memory.diary_engine instead.
This module will be removed in a future version.
"""

import warnings

from src.meta_learning.memory.diary_engine import ExperienceDiaryEngine


# Create compatibility alias with deprecation warning
class ExperienceDiary(ExperienceDiaryEngine):
    """
    DEPRECATED: Compatibility alias for ExperienceDiaryEngine.
    
    Please use ExperienceDiaryEngine from src.meta_learning.memory.diary_engine instead.
    This class will be removed in a future version.
    """

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "ExperienceDiary is deprecated and will be removed in a future version. "
            "Use ExperienceDiaryEngine from src.meta_learning.memory.diary_engine instead.",
            DeprecationWarning,
            stacklevel=2
        )
        super().__init__(*args, **kwargs)


__all__ = ['ExperienceDiary']
