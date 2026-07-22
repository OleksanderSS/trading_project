# Configuration System (`src/config`)

This directory contains the project's **Single Source of Truth (Stage 0)**, serving as the central nervous system for all operational settings.

## Core Philosophy: Centralization and Simplicity

The entire configuration is managed by the `UnifiedConfigManager`, which centralizes all logic for loading, merging, and accessing configuration parameters. This approach ensures that every module in the pipeline (Stages 1-7) draws from the same consistent data source.

## Key Components

### 1. `unified_config_manager.py`

This is the heart of the configuration system. It is a singleton class responsible for the core logic of loading, validating, and distributing settings throughout the application.

**Key Responsibilities:**

-   **Loads YAML Files**: Automatically scans the `src/config` directory and loads all `.yaml` files.
-   **Environment Merging**: Merges environment-specific configurations over the base configuration for flexible overrides.
-   **Secret Resolution**: Securely resolves secrets by replacing keys ending in `_env` with values from the `SecretsManager`.
-   **Dynamic Rule Integration**: Supports the dynamic loading of `generated_context_rules.yaml` to adapt the system to changing market regimes.
-   **Dynamic Access**: Provides convenient dot-notation access to parameters (e.g., `config.database.host`).

### 2. Modular YAML Configuration Files (`*.yaml`)

Settings are organized by pipeline stage and specific features, allowing for 'No-Code' adjustments to the trading system:

-   **Assets & Data**: `assets.yaml` (tickers/timeframes), `collectors.yaml` (Stage 1), and `data_sources.yaml`.
-   **Processing & Features**: `processing.yaml` (Stage 2), `features.yaml` (Stage 3), and `targets.yaml`.
-   **Modeling & Strategy**: `models.yaml` (Stage 4) and `strategy.yaml` (Stage 6).
-   **Analysis & System**: `analysis.yaml` (Stage 7), `system.yaml`, and `error_handling.yaml`.

## How to Use

To access the configuration from anywhere in the application, simply call the `get_current_config()` function:

```python
from src.config.unified_config_manager import get_current_config

# Get the singleton instance of the config manager
config = get_current_config()

# Access parameters using dot notation
host = config.database.host
api_key = config.collectors.my_api.api_key
```

## Refactoring Summary

This directory has been refactored to centralize all configuration logic. The original Python-based configuration files (`feature_config.py`, `analysis_config.py`) have been replaced with modular YAML files and the `UnifiedConfigManager`. This new architecture is simpler, more robust, and significantly easier to maintain.