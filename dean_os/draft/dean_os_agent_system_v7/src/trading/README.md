# Trading Module (`src/trading`)

This module serves as the primary engine for **Stage 6 (Signal Execution & Portfolio Tracking)**. It bridges the gap between model predictions and market participation by transforming forecasts into actionable trade orders while maintaining strict risk controls.

## Core Components

1.  **`consensus_engine.py`**: The "Decision Maker" of the system. It merges multi-model predictions and historical context (from the Experience Diary) into a single, high-confidence trade signal.
2.  **`virtual_portfolio.py`**: A local, persistent tracker for Paper Trading. It provides real-time tracking of cash, equity, and PnL, ensuring the system's state is preserved across sessions.
3.  **`portfolio_manager.py`**: The "Risk Officer." It implements position limits, stop-losses, take-profits, and dynamic risk rules to protect the capital.
4.  **`trader.py`**: The "Executioner." It provides the interface for live order execution, handling communication with external brokers (e.g., Alpaca, Interactive Brokers).

## Risk Management & Security

The system includes a dedicated security layer to protect capital during unforeseen market events:
- **Kill-Switch**: An automated emergency mechanism that suspends all trading activities if daily or weekly loss thresholds are breached.
- **Exposure Limits**: Enforces strict diversification rules at both the individual ticker level and the broader sector level to prevent over-concentration.
- **Safety Protocol**: Seamless integration with the `ConsensusEngine` to block trade execution during periods of extreme market volatility or when system-wide kill-switch triggers are active.

## Integration

The `trading` module is deeply integrated into the 8-stage pipeline:
*   **Input**: Receives forecasts from **Stage 5 (Prediction)**.
*   **Optimization**: Collaborates with **`src/optimization/portfolio_optimizer.py`** to calculate optimal asset weights and position sizes based on modern portfolio theory.
*   **Output**: Generates execution logs and portfolio states for **Stage 7 (Evaluation)**.

This architecture ensures that every trade is verified for risk, optimized for size, and tracked for performance in a unified workflow.