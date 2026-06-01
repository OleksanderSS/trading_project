"""
Specialized analyzer for portfolio concentration risk.
Calculates HHI, diversification ratios, and sector exposure.
"""

from typing import Any

import numpy as np

from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger("ConcentrationAnalyzer")


class ConcentrationAnalyzer:
    def __init__(self, max_asset_weight: float = 0.25):
        self.max_asset_weight = max_asset_weight

    def analyze(self, positions: dict[str, dict[str, Any]]) -> dict[str, Any]:
        """Calculates concentration metrics for the current portfolio."""
        if not positions:
            return {"status": "empty_portfolio", "hhi": 0}

        values = [p.get("current_value", 0) for p in positions.values()]
        total_value = sum(values)
        if total_value == 0:
            return {"status": "zero_value", "hhi": 0}

        weights = np.array([v / total_value for v in values])

        # Herfindahl-Hirschman Index (HHI)
        hhi = float(np.sum(weights**2))

        # Concentration Ratio (Top 1)
        max_weight = float(np.max(weights))

        breaches = []
        for symbol, pos in positions.items():
            weight = pos.get("current_value", 0) / total_value
            if weight > self.max_asset_weight:
                breaches.append({"symbol": symbol, "weight": weight, "limit": self.max_asset_weight})

        return {
            "status": "success",
            "hhi": hhi,
            "max_weight": max_weight,
            "is_diversified": hhi < 0.15,  # Industry standard for diversified
            "breaches": breaches,
            "effective_number_of_assets": float(1 / hhi) if hhi > 0 else 0,
        }
