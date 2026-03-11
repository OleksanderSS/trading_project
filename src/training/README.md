# Model Training Module (`src/training`)

This directory contains the core logic for the **Stage 4 (Modeling)** phase of the trading pipeline. It transforms engineered features into predictive intelligence through a structured 3-level training hierarchy, ensuring a balance between deep historical learning and immediate market reactivity.

## The 3-Level Training Hierarchy

The module is organized into three distinct layers to manage complexity and scale:

### 1. Strategy Level (Context-Aware Planning)
- **`adaptive_training_manager.py`**: The "brain" of the training process. It analyzes ticker sets, data quality, and market context to generate optimal training plans. **(Note: The current implementation is a high-level prototype that simulates the planning process and requires integration with real data sources and target generation systems).**

### 2. Orchestration Level (Execution & Integration)
- **`unified_training_manager.py`**: The primary entry point for executing training cycles. It orchestrates the lifecycle defined by the Strategy Level, managing:
    - **Arena Integration**: Running model "battles" to select the best architecture.
    - **Colab Sync**: Offloading heavy deep-learning tasks to GPU-enabled environments.
    - **Lifecycle Management**: Handling model versioning, state persistence, and performance tracking.

### 3. Worker Level (Specialized Training Engines)
- **`batch_trainer.py`**: Optimized for high-scale operations, managing the simultaneous training of multiple models in parallel batches.
- **`progressive_trainer.py`**: Implements online and incremental learning, updating model weights with new data points without full rebuilds.
- **`light_model_trainer.py`**: A high-speed engine specialized for GBDT and ensemble methods (LightGBM, XGBoost, CatBoost, and RandomForest).

## Role in the Pipeline

As the core of **Stage 4**, this module acts as the "skill development" center. It takes cleaned data and features (Stages 1-3) and produces validated, "champion" models ready for inference in **Stage 5 (Prediction)**. By separating strategy from execution, the system can scale from a single ticker to thousands of assets while maintaining model integrity and resource efficiency.