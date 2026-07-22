# Metrics Module (`src/metrics`)

This module serves as the **Source of Truth for Evaluation** across the entire project, primarily supporting **Stage 4 (Modeling)** and **Stage 7 (Evaluation)** of the pipeline.

## Overview
The Metrics module centralizes all mathematical formulas used to quantify the performance of machine learning models and trading strategies. By providing a unified interface, it ensures that a model's success is measured identically during the training phase (to select champions) and the final evaluation phase (to assess real-world viability).

## Core Components

1.  **`model_metrics.py`**: Contains standard Machine Learning evaluation metrics.
    *   **Classification**: Accuracy, Precision, Recall, F1-Score, and Log-Loss.
    *   **Regression**: RMSE (Root Mean Squared Error), MAE (Mean Absolute Error), and R2-Score.
2.  **`financial_metrics.py`**: Implements professional trading and risk management metrics.
    *   **Risk/Reward**: Sharpe Ratio, Sortino Ratio, and Calmar Ratio.
    *   **Drawdown Analysis**: Maximum Drawdown (MDD), Drawdown Duration, and Recovery Factor.
    *   **Execution**: Win Rate, Profit Factor, and Expectancy.
3.  **`calculator.py`**: The primary unified interface.
    *   Provides the `MetricsCalculator` class which can compute a comprehensive "performance snapshot" (both ML and Financial) in a single call.
4.  **`metrics_utils.py`**: Supporting utilities for time-series adjustments, such as annualizing returns and handling volatility scaling.

## Integration & Standards
*   **Pipeline Integration**: Used in **Stage 4** to rank models in the Arena and in **Stage 7** to generate final performance reports for the user.
*   **Accuracy**: All formulas strictly follow **industry-standard financial mathematics** (e.g., using log-returns where appropriate and standard risk-free rate assumptions).
*   **Consistency**: By centralizing these calculations, the system prevents "metric drift" where different modules might otherwise use slightly different formulas for the same KPI.