"""
Real-time Context Awareness - Facade for Modular Context Engine.
Maintains backward compatibility with the original ContextAwarenessEngine.
"""

from .context.manager import ContextAwarenessEngine as ModularContextAwarenessEngine


class ContextAwarenessEngine(ModularContextAwarenessEngine):
    """
    Facade for ContextAwarenessEngine.
    Delegates to modular components in the 'context' subdirectory.
    """
    pass
