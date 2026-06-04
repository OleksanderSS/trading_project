#!/usr/bin/env python3
"""
Weight Stability Monitor - Facade for the modular Weight-Stability System.
This module maintains backward compatibility while delegating to the new modular structure.
"""

from datetime import datetime
from typing import Any

from .weight_stability.manager import WeightStabilityMonitor as ModularWeightStabilityMonitor


class WeightStabilityMonitor(ModularWeightStabilityMonitor):
    """
    Facade for WeightStabilityMonitor.
    Maintains the original API but uses modular components internally.
    """
    pass

def get_weight_stability_monitor(stability_threshold: float = 0.1,
                               window_size: int = 10,
                               max_change_per_update: float = 0.15) -> WeightStabilityMonitor:
    """Factory function to get WeightStabilityMonitor instance."""
    return WeightStabilityMonitor(stability_threshold, window_size, max_change_per_update)

def monitor_weight_stability_quick(new_weights: dict[str, float],
                                 last_weights: dict[str, float],
                                 stability_threshold: float = 0.1) -> dict[str, Any]:
    """
    Quick weight stability monitoring.

    Args:
        new_weights: New weight dictionary
        last_weights: Previous weight dictionary
        stability_threshold: Threshold for stability alerts

    Returns:
        Stability monitoring result dictionary
    """
    monitor = get_weight_stability_monitor(stability_threshold)
    # Note: Modular version stores state internally,
    # but quick monitor usually expects stateless or one-off use.
    # To maintain consistency with original intent, we update it once.
    monitor.current_weights = last_weights
    return monitor.update_weights(new_weights, datetime.now())
