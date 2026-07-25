"""Unit tests for AdvancedBacktestEngine and related classes."""

import pandas as pd


class TestTransactionCostModel:
    """Tests for TransactionCostModel in src/backtesting/advanced/advanced_engine.py"""

    def test_calculate_execution_costs_returns_dict(self):
        """Test that calculate_execution_costs returns expected dict structure."""
        from src.backtesting.advanced.advanced_engine import TransactionCostModel

        model = TransactionCostModel()
        result = model.calculate_execution_costs(
            trade_value=100000,
            daily_volume=1000000,
            volatility=0.02
        )

        assert 'commission' in result
        assert 'spread' in result
        assert 'market_impact' in result
        assert 'slippage' in result
        assert 'total' in result
        assert 'total_pct' in result
        assert result['total'] > 0

    def test_calculate_execution_costs_with_order_size(self):
        """Test with explicit order_size_pct."""
        from src.backtesting.advanced.advanced_engine import TransactionCostModel

        model = TransactionCostModel()
        result = model.calculate_execution_costs(
            trade_value=50000,
            daily_volume=1000000,
            volatility=0.01,
            order_size_pct=0.05
        )

        assert result['total'] > 0
        assert isinstance(result['total'], float)


class TestBiasDetector:
    """Tests for BiasDetector in src/backtesting/advanced/advanced_engine.py"""

    def test_detect_look_ahead_bias_no_common_columns(self):
        """Test look-ahead bias detection with no common columns."""
        from src.backtesting.advanced.advanced_engine import BiasDetector

        detector = BiasDetector()
        signals = pd.DataFrame({'AAPL': [1, 2, 3], 'MSFT': [1, 2, 3]})
        future_prices = pd.DataFrame({'GOOG': [1, 2, 3]})

        result = detector.detect_look_ahead_bias(signals, future_prices)

        assert result['has_look_ahead_bias'] == False
        assert 'suspicious_signals' in result

    def test_detect_survivorship_bias_with_delisted(self):
        """Test survivorship bias detection with delisted tickers."""
        from src.backtesting.advanced.advanced_engine import BiasDetector

        detector = BiasDetector()
        historical = ['AAPL', 'MSFT', 'GOOG', 'DELISTED1', 'DELISTED2']
        current = ['AAPL', 'MSFT', 'GOOG']

        result = detector.detect_survivorship_bias(historical, current)

        assert result['potential_bias'] is True
        assert result['missing_assets_count'] == 2
        assert 'DELISTED1' in result['missing_assets']
        assert 'DELISTED2' in result['missing_assets']


class TestWalkForwardOptimizer:
    """Tests for WalkForwardOptimizer in src/backtesting/advanced/advanced_engine.py"""

    def test_evaluate_parameters_with_valid_data(self):
        """Test parameter evaluation with valid data."""
        from src.backtesting.advanced.advanced_engine import WalkForwardOptimizer

        optimizer = WalkForwardOptimizer()
        data = pd.DataFrame({
            'price': [100, 102, 101, 103, 105, 104, 106, 108, 107, 109]
        })

        result = optimizer._evaluate_parameters(data)

        assert 'return' in result
        assert 'sharpe' in result
        assert 'max_drawdown' in result

    def test_evaluate_parameters_with_empty_data(self):
        """Test parameter evaluation with empty data."""
        from src.backtesting.advanced.advanced_engine import WalkForwardOptimizer

        optimizer = WalkForwardOptimizer()
        result = optimizer._evaluate_parameters(None)

        assert result['return'] == 0.0
        assert result['sharpe'] == 0.0
        assert result['max_drawdown'] == 0.0

    def test_calculate_average_performance_empty(self):
        """Test average performance with empty results."""
        from src.backtesting.advanced.advanced_engine import WalkForwardOptimizer

        optimizer = WalkForwardOptimizer()
        assert optimizer._calculate_average_performance([]) == {}


class TestAdvancedBacktestEngine:
    def test_backtest_uses_signals_instead_of_buy_and_hold_average(self):
        from src.backtesting.advanced.advanced_engine import AdvancedBacktestEngine

        prices = pd.DataFrame(
            {
                "AAPL": [100.0, 110.0, 121.0],
                "MSFT": [100.0, 90.0, 81.0],
            },
            index=pd.date_range("2024-01-01", periods=3),
        )
        long_winner = pd.DataFrame(
            {"AAPL": [1, 1, 1], "MSFT": [0, 0, 0]},
            index=prices.index,
        )
        long_loser = pd.DataFrame(
            {"AAPL": [0, 0, 0], "MSFT": [1, 1, 1]},
            index=prices.index,
        )

        engine = AdvancedBacktestEngine()
        winner_report = engine.run_comprehensive_backtest(
            prices,
            long_winner,
            {"initial_capital": 100.0, "slippage_adjustment": False, "bias_detection": False},
        )
        loser_report = engine.run_comprehensive_backtest(
            prices,
            long_loser,
            {"initial_capital": 100.0, "slippage_adjustment": False, "bias_detection": False},
        )

        assert winner_report["performance_metrics"]["total_return"] > 0
        assert loser_report["performance_metrics"]["total_return"] < 0
