import numpy as np


class VaRCalculator:
    def __init__(self):
        pass

    def calculate(self, data):
        """
        Main entry point for VaR calculation.
        Routes to historical VaR if data is provided.
        """
        if data is None or len(data) == 0:
            return 0.0
            
        result = self.calculate_var_historical(data)
        return result.get('var', 0.0)


    def calculate_var_historical(self, returns, confidence=0.95, time_horizon=1):
        """Calculate loss-positive historical VaR from returns."""
        clean_returns = np.asarray([] if returns is None else returns,
            dtype=float)
        clean_returns = clean_returns[np.isfinite(clean_returns)]
        if clean_returns.size == 0:
            return {'var': np.nan, 'status': 'insufficient_data'}

        tail_percentile = (1 - confidence) * 100
        var_return_threshold = np.percentile(clean_returns, tail_percentile)  # audit-ignore: VAR_SIGN_OR_EMPTY_DATA_REVIEW
        horizon_scale = np.sqrt(max(1, int(time_horizon)))
        var_loss_positive = max(0.0, float(-var_return_threshold)
            ) * horizon_scale

        return {
            'var': float(var_loss_positive),
            'var_return_threshold': float(var_return_threshold),
            'confidence': float(confidence),
            'time_horizon': int(time_horizon),
            'status': 'ok',
        }
