from typing import Any

import numpy as np

from src.core.exceptions import DataProcessingError
from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger("RegimeStabilityAnalyzer")


class RegimeStabilityAnalyzer:
    """Аналізує стабільність режимів та частоту їх зміни."""

    @staticmethod
    def get_most_frequent_switch(switches: list[dict[str, Any]]) ->dict[str, Any]:
        """Визначає найбільш частий тип зміни режиму."""
        if not switches:
            return {}
        try:
            switch_counts = {}
            for switch in switches:
                switch_key = f"{switch['from_regime']}->{switch['to_regime']}"
                switch_counts[switch_key] = switch_counts.get(switch_key, 0
                    ) + 1
            most_frequent = max(switch_counts.items(), key=lambda x: x[1])
            from_regime, to_regime = most_frequent[0].split('->')
            return {'from_regime': from_regime, 'to_regime': to_regime,
                'count': most_frequent[1]}
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            logger.error(f"Error determining most frequent switch: {e}", exc_info=True)
            raise DataProcessingError("Could not determine most frequent switch") from e

    @staticmethod
    def calculate_average_stable_period(switches: list[dict[str, Any]]) ->float:
        """Розраховує середній період стабільності режиму (в годинах)."""
        try:
            if len(switches) < 2:
                return float('inf')
            stable_periods = []
            for i in range(1, len(switches)):
                prev_switch_time = switches[i - 1]['timestamp']
                curr_switch_time = switches[i]['timestamp']
                stable_period = (curr_switch_time - prev_switch_time
                    ).total_seconds() / 3600
                stable_periods.append(stable_period)
            return float(np.mean(stable_periods)) if stable_periods else float(
                'inf')
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            logger.error(f"Error calculating average stable period: {e}", exc_info=True)
            raise DataProcessingError("Could not calculate average stable period") from e

    @staticmethod
    def calculate_regime_stability(records: list[dict[str, Any]]) ->float:
        """Розраховує загальний показник стабільності режиму."""
        if not records or len(records) < 2:
            return 1.0
        switches = sum(1 for i in range(1, len(records)) if records[i][
            'regime'] != records[i - 1]['regime'])
        return float(1.0 - switches / len(records))
