"""Compatibility import for the canonical analyst-core reasoning snapshot."""

from dean_os.analyst_core.analyst_core_reasoning_snapshot import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_RUNTIME_JSON,
    SNAPSHOT_CONTRACT,
    AnalystCoreReasoningSnapshot,
    render_reasoning_snapshot_markdown,
)

__all__ = [
    "AnalystCoreReasoningSnapshot",
    "DEFAULT_OUTPUT_DIR",
    "DEFAULT_RUNTIME_JSON",
    "SNAPSHOT_CONTRACT",
    "render_reasoning_snapshot_markdown",
]
