# Simulation Module (`src/simulation`)

This module provides tools for advanced strategy validation and risk assessment through realistic market replication and sensitivity analysis. Its primary purpose is to provide a robust environment for offline testing before any strategy is deployed to live or paper trading.

## Role in the Pipeline
Simulations act as a final "sanity check" in **Stage 7 (Advanced Risk & Sensitivity Analysis)**. While standard backtesting shows how a strategy performed on *actual* history, the simulation module tests how it might perform across a wide range of *statistically possible* futures and under realistic market frictions.

To ensure result consistency across the entire system, the simulation engine must use the same `VirtualPortfolio` and `MetricsCalculator` as the trading and backtesting modules.

## Key Components

### `simulation_engine.py`
The core engine responsible for executing complex simulations to evaluate portfolio and strategy performance under diverse conditions:

*   **Market Frictions:** Simulates realistic execution challenges like slippage, latency, and partial fills to account for real-world trading costs.
*   **Monte Carlo Simulations:** Generates thousands of potential future paths for asset prices to assess portfolio risk, probability of ruin, and expected return distributions.
*   **Stress Testing:** Evaluates how the trading system and portfolio would behave during extreme market events (e.g., Flash Crashes, Macro Shocks).
*   **Bootstrapping:** Uses historical return data to create synthetic datasets, allowing for the calculation of confidence intervals for performance metrics like Sharpe Ratio and Maximum Drawdown.
*   **Scenario Analysis:** Enables "What If" testing by manually defining specific market environments to observe strategy adaptability.

## Workflow Integration
After the standard backtest is complete, the simulation module is triggered to perform stress-testing and sensitivity analysis, ensuring the strategy is not overfit to a single historical path and can withstand execution anomalies.