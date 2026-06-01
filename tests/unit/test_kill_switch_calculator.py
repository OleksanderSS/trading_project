import pytest
import pandas as pd
from src.risk.kill_switch.calculator import KillSwitchCalculator
from src.core.exceptions import DataProcessingError
from unittest.mock import MagicMock

def test_calculate_risk_metrics_raises_exception_on_invalid_data():
    """Verify that KillSwitchCalculator raises DataProcessingError when calculation fails."""
    config_mock = MagicMock()
    calculator = KillSwitchCalculator(config_manager=config_mock)
    
    # Passing invalid market_data to trigger an exception
    with pytest.raises(DataProcessingError):
        calculator.calculate_risk_metrics({}, None, 'normal')

def test_calculate_portfolio_metrics_raises_exception_on_invalid_data():
    """Verify that KillSwitchCalculator raises DataProcessingError in portfolio metrics."""
    config_mock = MagicMock()
    calculator = KillSwitchCalculator(config_manager=config_mock)
    
    with pytest.raises(DataProcessingError):
        # Invalid market_data
        calculator.calculate_portfolio_metrics({'AAPL': {'current_value': 100}}, None)
