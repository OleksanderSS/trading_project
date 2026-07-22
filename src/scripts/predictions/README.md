# Predictions Module (`src/predictions`)

This module serves as the primary engine for **Stage 5 (Prediction & Confidence Calculation)** of the trading pipeline. It is responsible for transforming raw model outputs into actionable forecasts.

## Role in Pipeline
As the core of **Stage 5**, this module takes trained models (from Stage 4) and processed features (from Stage 3) to generate time-series forecasts. It ensures that every prediction is not just a raw number, but a contextualized estimate with an associated confidence score.

## Core Components

1.  **`prediction_utils.py`**: 
    *   **Standardization**: Ensures all model outputs follow a unified schema.
    *   **Inverse Scaling**: Automatically reverses any feature scaling (e.g., MinMax, Standard) to return predictions to their original price or percentage units.
    *   **Classification Logic**: Handles the conversion of raw probabilities into discrete classes (Buy/Sell/Hold) based on dynamic thresholds.

2.  **`ExperienceDiary` Integration**:
    *   Integrates with `src/meta_learning/experience_diary.py` to adjust the **Confidence Score** of a prediction.
    *   If a specific model has historically performed poorly in the current market context (regime), the module penalizes its confidence score to prevent over-reliance on unreliable signals.

## Prediction Adjustment & Refinement
The module is responsible for **Prediction Adjustment**. It does not simply pass through model outputs; it refines them by:
*   Applying filters based on recognized **Market Patterns** (`src/patterns/`).
*   Adjusting forecasts based on the **Market Context** identified in Stage 2.
*   Merging multi-timeframe predictions into a coherent forecast for the `ConsensusEngine` in Stage 6.

---
*Note: This module ensures that "Raw Data -> Raw Model -> Refined Prediction" remains a clean, auditable flow.*