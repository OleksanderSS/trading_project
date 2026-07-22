"""Lazy exports for Stage 2 processing components.

Importing a small schema/normalization helper must not initialize cloud storage,
the full configuration audit, or the Stage 2 orchestrator.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    'ProcessingStage',
    'ProcessingValidator',
    'ProcessingDataHandler',
    'ProcessingStorage'
]


def __getattr__(name: str) -> Any:
    if name == "ProcessingDataHandler":
        from .data_handler import ProcessingDataHandler

        return ProcessingDataHandler
    if name == "ProcessingStage":
        from .orchestrator import ProcessingStage

        return ProcessingStage
    if name == "ProcessingStorage":
        from .storage import ProcessingStorage

        return ProcessingStorage
    if name == "ProcessingValidator":
        from .validator import ProcessingValidator

        return ProcessingValidator
    raise AttributeError(name)
