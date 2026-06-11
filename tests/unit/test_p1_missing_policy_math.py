import numpy as np
import pandas as pd


def test_algorithms_backtest_keeps_missing_price_gap_when_position_is_open():
    from src.algorithms.advanced_backtest_engine import AdvancedBacktestEngine

    engine = object.__new__(AdvancedBacktestEngine)
    prices = pd.DataFrame(
        {"AAPL": [100.0, 101.0, None, 103.0, 104.0]},
        index=pd.date_range("2024-01-01", periods=5),
    )
    signals = pd.DataFrame({"AAPL": [1, 1, 1, 1, 1]}, index=prices.index)

    equity = engine._simulate_returns(prices, 1000.0, signals)

    assert equity.iloc[0] == 1000.0
    assert pd.isna(equity.iloc[2])
    assert pd.isna(equity.iloc[3])


def test_algorithms_backtest_missing_price_gap_is_neutral_without_position():
    from src.algorithms.advanced_backtest_engine import AdvancedBacktestEngine

    engine = object.__new__(AdvancedBacktestEngine)
    prices = pd.DataFrame(
        {"AAPL": [100.0, 101.0, None, 103.0]},
        index=pd.date_range("2024-01-01", periods=4),
    )
    signals = pd.DataFrame({"AAPL": [0, 0, 0, 0]}, index=prices.index)

    equity = engine._simulate_returns(prices, 1000.0, signals)

    assert equity.notna().all()
    assert equity.tolist() == [1000.0] * 4


def test_stage7_backtest_keeps_missing_price_gap_when_position_is_open():
    from src.backtesting.advanced.advanced_engine import AdvancedBacktestEngine

    engine = AdvancedBacktestEngine()
    # Mock cost model as in original test
    engine.cost_model = type(
        "CostModel",
        (),
        {
            "commission_pct": 0.0,
            "spread_bps": 0.0,
            "slippage_pct": 0.0,
        },
    )()
    prices = pd.DataFrame(
        {"AAPL": [100.0, 101.0, None, 103.0, 104.0]},
        index=pd.date_range("2024-01-01", periods=5),
    )
    signals = pd.DataFrame({"AAPL": [1, 1, 1, 1, 1]}, index=prices.index)

    equity = engine._simulate_returns(prices, 1000.0, signals=signals)

    assert equity.iloc[0] == 1000.0
    assert np.isclose(equity.iloc[2], 1010.0)
    assert np.isclose(equity.iloc[3], 1010.0)
    assert np.isclose(equity.iloc[4], 1019.801980198)


def test_macro_score_scaling_preserves_missing_scores():
    from src.analytics.calculators.macro_score_calculator import MacroScoreCalculator

    calc = MacroScoreCalculator({"gdp": {"weight": 1.0, "direction": "positive"}})
    scaled = calc._scale_final_score(pd.Series([1.0, np.nan, 3.0]))

    assert scaled.iloc[0] == 0.0
    assert pd.isna(scaled.iloc[1])
    assert scaled.iloc[2] == 100.0


def test_macro_weighted_composite_uses_available_indicators_only():
    from src.analytics.calculators.macro_score_calculator import MacroScoreCalculator

    calc = MacroScoreCalculator(
        {
            "gdp": {"weight": 1.0, "direction": "positive"},
            "cpi": {"weight": 1.0, "direction": "negative"},
        }
    )
    scores = pd.DataFrame(
        {
            "gdp_score": [1.0, np.nan, np.nan],
            "cpi_score": [np.nan, -1.0, np.nan],
        }
    )

    composite = calc._calculate_weighted_composite(scores)

    assert composite.iloc[0] == 1.0
    assert composite.iloc[1] == -1.0
    assert pd.isna(composite.iloc[2])


def test_market_context_marks_missing_features_explicitly():
    from src.analytics.context.market_context_analyzer import MarketContextAnalyzer

    analyzer = MarketContextAnalyzer(["volatility_5d", "rsi_current"])
    data = pd.DataFrame(
        {
            "close": [100.0, None, 102.0],
            "rsi": [None, None, None],
        }
    )

    result = analyzer.analyze(data)

    assert "volatility_5d" in result["missing_context_features"]
    assert result["market_context_vector"]["volatility_5d"] == 0.0
    assert result["market_context_vector"]["rsi_current"] == 50.0


def test_market_regime_analyzer_drops_missing_returns_instead_of_zero_filling():
    from src.analytics.context.market_regime_analyzer import MarketRegimeAnalyzer

    class DetectorStub:
        def __init__(self):
            self.returns = None

        def detect_regime(self, returns, data_bundle=None):
            self.returns = returns
            return {"regime": "normal", "confidence": 0.9}

    prices = pd.Series(np.linspace(100.0, 140.0, 40))
    prices.iloc[10] = np.nan
    analyzer = MarketRegimeAnalyzer()
    analyzer._detector = DetectorStub()

    result = analyzer.analyze(pd.DataFrame({"close": prices}))

    assert result["regime"] == "normal"
    assert analyzer._detector.returns is not None
    assert np.isfinite(analyzer._detector.returns).all()
    assert not np.isclose(analyzer._detector.returns, 0.0).any()


def test_anomaly_detector_imputes_with_training_medians_not_zero():
    from src.analytics.detectors.anomaly_detector import AnomalyDetector

    detector = AnomalyDetector()
    detector.feature_columns = ["feature_a", "feature_b"]
    detector.feature_medians = pd.Series({"feature_a": 3.0, "feature_b": 30.0})
    features = pd.DataFrame({"feature_a": [np.nan, 5.0], "feature_b": [10.0, np.nan]})

    imputed = detector._impute_with_training_medians(features)

    assert imputed.loc[0, "feature_a"] == 3.0
    assert imputed.loc[1, "feature_b"] == 30.0


def test_market_regime_metrics_drop_missing_returns_instead_of_zero_filling():
    from src.analytics.utils.analytics_math import calculate_market_regime_metrics

    prices = pd.Series([100.0, 101.0, None, 103.0, 104.0, 105.0])

    metrics = calculate_market_regime_metrics(prices, window=2)
    valid_returns = prices.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).dropna()
    expected_volatility = valid_returns.rolling(2, min_periods=1).std().shift(1).iloc[-1]
    expected_trend = valid_returns.rolling(2, min_periods=1).mean().shift(1).iloc[-1]

    assert np.isclose(metrics["volatility"], expected_volatility)
    assert np.isclose(metrics["trend"], expected_trend)


def test_diversification_ratio_drops_missing_return_rows():
    from src.analytics.utils.analytics_math import calculate_diversification_ratio

    returns = pd.DataFrame(
        {
            "AAA": [0.01, np.nan, 0.02, -0.01],
            "BBB": [0.02, 0.01, np.nan, -0.02],
        }
    )
    weights = np.array([0.5, 0.5])

    ratio = calculate_diversification_ratio(returns, weights)
    clean_returns = returns.dropna(how="any")
    portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(clean_returns.cov(), weights)))
    weighted_vol = np.sum(weights * clean_returns.std())

    assert np.isclose(ratio, weighted_vol / portfolio_vol)


def test_management_data_cleaner_preserves_missing_numeric_values():
    from src.data.management.data_cleaner import DataCleaner

    df = pd.DataFrame(
        {
            "price": [1.0, np.inf, np.nan, -np.inf],
            "label": ["a", "b", "c", "d"],
        }
    )

    cleaned = DataCleaner.clean_numeric_data(df)

    assert cleaned["price"].isna().tolist() == [False, True, True, True]
    assert cleaned["label"].tolist() == ["a", "b", "c", "d"]


def test_synthetic_features_require_real_history_for_return_windows():
    from src.data.synthetic.data_generator import DataGenerator

    generator = DataGenerator(config_manager=None)
    full_index = pd.date_range(start="2020-01-01", end="2023-12-31", freq="1h")
    x = np.arange(len(full_index))
    close = pd.Series(100.0 + (0.01 * x) + np.sin(x / 6.0), index=full_index)
    generator._generate_price_series = lambda n_points: close.iloc[:n_points].to_numpy()

    features = generator.generate_synthetic_features()
    first_idx = features.index[0]
    expected_24h_returns = close.pct_change(24, fill_method=None)

    assert first_idx >= full_index[24]
    assert np.isclose(features.loc[first_idx, "returns_24h"], expected_24h_returns.loc[first_idx])


def test_drawdown_underwater_duration_handles_initial_boundary_without_fillna_zero():
    from src.analytics.calculators.drawdown_calculator import DrawdownCalculator

    df = pd.DataFrame(
        {
            "high": [100.0, 105.0, 105.0, 110.0],
            "close": [100.0, 104.0, 103.0, 111.0],
        }
    )

    duration = DrawdownCalculator.calculate_underwater_duration(df)

    assert duration.tolist() == [0, 1, 2, 0]


def test_feature_preparation_uses_median_imputation_not_zero_fill():
    from src.utils.feature_preparation import prepare_features_for_training

    df = pd.DataFrame(
        {
            "ticker": ["AAPL", "AAPL", "AAPL"],
            "feature_a": [1.0, None, 3.0],
            "feature_b": [float("inf"), 2.0, 4.0],
        }
    )

    features, columns = prepare_features_for_training(df, fill_na=True)

    assert columns == ["feature_a", "feature_b"]
    assert features["feature_a"].tolist() == [1.0, 2.0, 3.0]
    assert features["feature_b"].tolist() == [3.0, 2.0, 4.0]
