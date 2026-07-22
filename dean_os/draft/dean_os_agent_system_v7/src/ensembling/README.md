# Ensembling Module (`src/ensembling`)

The **Ensembling Module** serves as the **Strategic Decision Merger** for the system, operating as the final processing layer between **Stage 5 (Prediction)** and **Stage 6 (Signal Generation)**. Its primary mission is to aggregate the diverse "opinions" of multiple machine learning models into a single, high-confidence decision that outperforms any individual architecture.

### 1. Core Components

- **`ensemble.py`**: The mathematical core of the module. It features the **`StackedEnsemble`** class, which implements advanced Meta-Modeling. This logic trains a secondary model to learn how to optimally combine base model predictions based on current market conditions.
- **`ensemble_model.py`**: Provides standard ensemble architectures, including **Voting** (Hard/Soft), **Averaging**, and **Weighted** combinations, ensuring compatibility with the standard model interface.

### 2. Strategic Connectivity

The ensembling layer is deeply integrated into the system's "nervous system":

- **Model Inputs**: Receives raw outputs and logits from various architectures in **`src/models/`** (LSTM, XGBoost, Transformers, etc.) via the **`src/factories/model_factory.py`**.
- **Dynamic Weighting**: Interacts with **`src/meta_learning/experience_diary.py`** to adjust the influence of specific models in real-time. It queries the diary to see which models have historically excelled in the current market regime.
- **Signal Delivery**: Provides the final, refined consensus signals and confidence intervals to **`src/trading/consensus_engine.py`** for portfolio execution.

### 3. Goals & Robustness

The primary objective of this module is to improve prediction stability across different market regimes:

- **Variance Reduction**: By averaging or stacking predictions, the system cancels out individual model noise and prevents over-reliance on a single architecture.
- **Regime Adaptation**: Dynamically shifts capital allocation towards models that are currently "in-favor" based on the **Market Phase Analyzer** (Stage 2).
- **Robustness**: Ensures that the final signal is only generated when a strategic consensus is reached between diverse model types.

### 4. Operational Flow

1.  **Input**: N raw predictions and confidence scores gathered from the Stage 5 prediction loop.
2.  **Context Check**: Consult the Experience Diary for historical performance fingerprints in the current context.
3.  **Aggregation**: The **`StackedEnsemble`** applies weighted logic to produce a unified, context-aware forecast.
4.  **Output**: Final consensus signal and a refined confidence metric passed to Stage 6 for trade execution.