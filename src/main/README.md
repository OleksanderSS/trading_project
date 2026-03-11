# Main Module

This module serves as the **Central Control Center (Application Entry Point)** for the trading system, acting as the primary hub that initializes and coordinates the entire software lifecycle.

## Components

- **`trading_orchestrator.py`**: The root of the application, responsible for system-wide initialization, including logging, configuration management, and database connections. It acts as the "conductor" that prepares the environment for execution.
- **`cli.py`**: Handles command-line arguments and facilitates mode selection, providing a structured interface for user interaction with the system's various capabilities.
- **`modes/`**: A collection of operational scenarios that define how the system should behave:
  - **`train.py`**: Dedicated mode for model training and fine-tuning.
  - **`backtest.py`**: Executes historical simulations to validate strategies.
  - **`intelligent.py`**: The main scenario for Live/Paper trading, incorporating real-time context awareness.
  - **`monster_test.py`**: Performs comprehensive stress testing of the system under extreme data volumes.
  - **`web_ui.py`**: Entry point for launching the system's web-based dashboard and reporting interface.
  - **`training_data_pipeline.py`**: Specifically manages data flows required for training cycles.

Each mode in this directory is designed to trigger and manage a specific subset of **src/pipeline/pipeline_orchestrator.py** stages (0-7), ensuring that the system's logic is applied correctly to the chosen operational task.

The `main` module is where the 'User Interface' (whether CLI or Web-based) connects to the 'Logical Pipeline', ensuring that complex interactions between data, models, and execution are handled in a robust and controllable environment.