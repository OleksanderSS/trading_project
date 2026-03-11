# Pipeline (src/pipeline)

This module is the core of **Project Orchestration & Lifecycle Management**. It acts as the **"spine"** of the entire system, coordinating the data and model lifecycles from initial setup to final evaluation.

## Key Components

### `pipeline_orchestrator.py`
The main entry point and "conductor" of the system. It is responsible for:
- Orchestrating the **8 stages (0-7)** in the correct sequence.
- **State Transfer & Management**: Passing the `data` dictionary between stages.
- **Memory Cleanup**: Proactively clearing temporary data to optimize RAM usage during heavy training cycles.
- **Execution Profiling**: Logging performance, duration, and memory delta for each stage.

### `stages/`
Contains the actual implementation for each stage of the lifecycle. Every stage inherits from the **`BaseStage`** abstract class to ensure a unified `execute()` interface and standard error handling.

*   **Stage 0 (Setup)**: Environment initialization and configuration validation.
*   **Stage 1 (Collection)**: Raw data gathering from various APIs and sources.
*   **Stage 2 (Processing)**: Data sanitization, sampling, and market context analysis.
*   **Stage 3 (Features)**: Feature engineering, NLP processing, and enrichment.
*   **Stage 4 (Modeling)**: Model training, Bayesian optimization, and Arena selection.
*   **Stage 5 (Prediction)**: Generating forecasts and calculating confidence levels.
*   **Stage 6 (Trading)**: Consensus signals, risk filtering, and portfolio optimization.
*   **Stage 7 (Evaluation)**: Performance metrics, backtesting, and automated reporting.

### Checkpoint & Resume Logic
The orchestrator implements a robust **"Checkpoint & Resume"** mechanism. Each stage can save its output state to disk, allowing the pipeline to:
- Recover from unexpected failures without restarting the entire process.
- Skip resource-intensive stages (like Collection or Training) if valid checkpoints exist.
- Facilitate iterative testing of specific pipeline segments.

## Usage
The pipeline is the primary execution engine for the system, typically triggered via `src/main/trading_orchestrator.py` or the CLI. It adheres strictly to the configuration defined in `src/config/unified_config.yaml`.