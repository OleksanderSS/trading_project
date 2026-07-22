# Stage 3: Targets

This module is responsible for the definition and generation of machine learning targets (labels). It marks **Stage 3** of the production pipeline, where we define exactly *what* our models should learn to predict.

This module follows the project's standard architecture: **Orchestrator + Modular Components**, driven by a central configuration file.

## Core Components

### 1. `target_orchestrator.py`

The central engine for target generation. Its responsibilities are:
- **Reading Configuration**: It loads the list of desired targets from `src/config/targets.yaml`.
- **Delegating Tasks**: It does not perform calculations itself. Instead, it dynamically selects and invokes the appropriate "Calculator" for each target defined in the config.
- **Data Integration**: It appends the generated target columns to the main DataFrame.

### 2. `src/config/targets.yaml`

The "menu" of all possible target variables. This file defines which targets to generate, what type they are, and what parameters to use. This approach allows for easy experimentation and modification of targets without changing any code.

Example `targets.yaml` entry:
```yaml
  - name: "TARGET_CLASS_BINARY_UP_1_0.005"
    type: "classification_binary"
    params:
      description: "Price goes up by > 0.5% in the next 1 period?"
      base_col: "close"
      shift: -1
      threshold: 0.005
```

### 3. `calculators/` Directory

This directory contains the modular, specialized components that perform the actual calculations. Each calculator is a small class responsible for a specific type of target generation, as defined in the YAML config (`type` key).

- **`regression_calculator.py`**: Calculates continuous future values (e.g., future returns).
- **`classification_calculator.py`**: Generates binary (`0/1`) or multiclass (`0/1/2`) labels based on future price movements and thresholds.
- **`indicator_prediction_calculator.py`**: Creates targets by shifting an existing feature column into the future (e.g., predicting the value of RSI in the next period).

## Role in the Pipeline

The `TargetOrchestrator` is invoked in **Stage 3** of the main data processing pipeline, right after the `FeatureOrchestrator` (Stage 2) has enriched the dataset.

1.  **Input**: Receives a DataFrame with raw data and engineered features.
2.  **Process**: Groups data by `ticker` to prevent data leakage across assets. For each target in `targets.yaml`, it calls the corresponding calculator.
3.  **Output**: Returns the DataFrame with new target columns appended, ready for **Stage 4 (Model Training)**.

This architecture ensures that the process is robust, flexible, and strictly free of **lookahead bias**, as all target calculations are based on future data points relative to the current observation.
