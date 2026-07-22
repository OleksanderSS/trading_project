# Enrichment Module

The `src.features.enrichers` directory is a core component of the system's **Feature Engineering** pipeline. Its primary responsibility is to transform raw market and economic data into structured, predictive features suitable for machine learning models.

### Role in Feature Engineering

This module provides a suite of specialized "enrichers" that append various dimensions of data to the primary market time series:

*   **`technical_analysis_enricher.py`**: Calculates and adds standard Technical Analysis (TA) indicators, such as Relative Strength Index (RSI), Exponential Moving Averages (EMA), Bollinger Bands, and MACD.
*   **`macro_features_enricher.py`**: Harmonizes and integrates macroeconomic data (sourced from FRED, economic calendars, etc.) into the market data, aligning disparate frequencies (e.g., monthly GDP vs. daily prices).
*   **`time_features_enricher.py`**: Injects temporal context into the dataset, including seasonality markers, day of the week, hour of the day, and binary flags for market holidays or sessions.
*   **`significance_features_enricher.py`**: Identifies and highlights statistically significant events, such as abnormal price shocks, volume spikes, or volatility explosions relative to historical distributions.
*   **`derived_features_enricher.py`**: Generates high-order custom ratios and normalized metrics, such as Price-to-EMA distance, Volatility-to-Average-Volume ratios, and other cross-feature interactions.

### Architecture

To ensure a consistent and plug-and-playable pipeline:
*   **Unified Interface**: All enrichers inherit from the `BaseEnricher` abstract class.
*   **`enrich` Method**: Every module implements a standard `enrich(df, **kwargs)` method, allowing the `FeatureOrchestrator` to chain multiple enrichment steps seamlessly.
*   **Immutability**: Enrichers are designed to return a new or modified DataFrame, preserving the integrity of the original data throughout the transformation process.