
class WeightStabilityConfig:
    """Configuration and thresholds for the Weight Stability Monitor."""

    def __init__(self,
                 stability_threshold: float = 0.1,
                 window_size: int = 10,
                 max_change_per_update: float = 0.15):
        self.stability_threshold = stability_threshold
        self.window_size = window_size
        self.max_change_per_update = max_change_per_update

        self.STABILITY_METRICS = {
            'volatility': {
                'description': 'Standard deviation of weight changes',
                'lower_better': True,
                'threshold': 0.1
            },
            'drift': {
                'description': 'Cumulative weight drift over time',
                'lower_better': True,
                'threshold': 0.2
            },
            'consistency': {
                'description': 'Weight consistency score',
                'higher_better': True,
                'threshold': 0.8
            },
            'reversal_frequency': {
                'description': 'Frequency of weight direction reversals',
                'lower_better': True,
                'threshold': 0.3
            }
        }
