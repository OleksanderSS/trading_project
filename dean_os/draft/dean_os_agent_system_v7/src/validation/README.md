# Validation Module

This module provides the core technical tools for ensuring the **Robustness** and statistical integrity of models. It serves as the primary technical gatekeeper in **Stage 7 (Evaluation)**, focusing on preventing overfitting and data leakage.

### Core Components

*   **`data_leakage_detector.py`**: A critical security tool that scans training datasets and features to detect if "future" information (look-ahead bias) has inadvertently leaked into the model's inputs.
*   **`time_series_validator.py`**: Implements specialized cross-validation schemes for non-shufflable financial time series, including Walk-forward and Purged K-Fold validation.
*   **`validation_protocols.py`**: Defines advanced statistical procedures such as **Combinatorial Purged Cross-Validation** to ensure consistent performance across diverse market slices.

### Strategic Note
While this module handles technical validation (leakage, stability), the business-level comparison and model selection (the "Battle" logic) have been moved to **`src/analytics/reporting/arena`** to better align with the reporting and decision-making flow.

### Purpose
The validation module is the gateway to production. It ensures that every model passing through the pipeline is statistically sound and free from the structural biases common in financial machine learning.