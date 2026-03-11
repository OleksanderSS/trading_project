# src/optimization

This folder contains the logic responsible for **Stage 6 (Capital Allocation & Portfolio Optimization)**: converting analytical outputs into executable portfolio allocations. This is where **'predictions'** become **'trades'**.

### Core Components

*   **`portfolio_optimizer.py`**: The primary engine that implements mathematical allocation models to transform raw signals and predictions into optimal asset weights.
    *   **Mean-Variance Optimization (MPT)**: Finds the Efficient Frontier, including Maximum Sharpe Ratio and Minimum Volatility portfolios.
    *   **Risk-Parity Weighting**: Allocates capital based on the risk contribution of each asset, ensuring a more balanced risk profile.
    *   **Black-Litterman Model Integration**: Combines market equilibrium with subjective "views" (derived from model predictions) to produce stable, intuitive asset allocations.
    *   **Risk Measurement**: Utilizes **src/analytics/calculators/** for consistent calculation of volatility, drawdowns, and other risk metrics across the system.

### Workflow

1.  **Input**: Receives signals from **src/trading/consensus_engine.py** and risk metrics (volatility, correlation data) from the **src.analytics** layer.
2.  **Optimization**: Runs the configured optimization algorithm (e.g., Quadratic Programming for MPT) to determine risk-based weights.
3.  **Constraint Enforcement**: Integrates with **src/trading/portfolio_manager.py** to apply regulatory and business constraints, such as maximum exposure per asset and sector limits.
4.  **Refinement**: Applies filters based on the `src.config.strategy.yaml` settings.
5.  **Output**: Produces a target weight vector for execution.