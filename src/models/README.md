# Unified Model Repository - Architecture Zoo & Selection Engine

This module serves as the central repository for all machine learning architectures and selection logic within the trading system. It acts as a **Unified Model Repository**, where every model in this directory is a pluggable component fully integrated into the 8-stage Pipeline via **`ModelFactory`** (Stage 4) and **`ModelSelector`** (Stage 5). This modular structure allows for seamless scaling and adding new model types without changing the core pipeline logic.

## 📁 Models Organization

### 1. Model Architectures
A comprehensive collection of model implementations categorized by their underlying technology:
- **`neural/`**: Deep Learning models like `lstm_model.py`, `gru_model.py`, `cnn_model.py`, `transformer_model.py`.
- **`tree/`**: Gradient Boosting and tree-based models like `catboost_model.py`, `xgboost_model.py`, `lightgbm_model.py`.
- **`linear/`**: Traditional ML models like `linear_model.py`, `svm_model.py`, `knn_model.py`.
- **`ensemble/`**: Logic for combining multiple models.

### 2. `dean/`
Integration logic for the **Distributed Evolutionary Network**. This sub-module contains the bootstrap and actor-critic logic used for self-improving systems, managed via `dean_integration.py` and `dean_trading_models.py`.

### 3. `model_selector/`
The "Intelligence" layer of **Stage 5**. It uses context-aware logic to dynamically switch between models or adjust ensemble weights based on the current market regime. Key components include:
- `intelligent_model_selector.py`
- `competence_analyzer.py`
- `dynamic_weight_selector.py`

### 4. `optimization/` (Note: Related logic may exist in `src/optimization`)
Dedicated logic for hyperparameter optimization. This module is heavily utilized in **Stage 4 (Modeling)** to ensure each architecture is tuned to its peak performance for specific tickers and timeframes.

## 🏗️ Management & Lifecycle

- **Instantiation**: All architectures are standardized and instantiated through the **`factory.py`** in this directory, ensuring a unified interface across the pipeline.
- **`trained/`** (Suggested): A potential future directory for storing serialized model artifacts (`.joblib`, `.pkl`, `.pt`). This would help maintain version control of active models.

## 🎯 Design Principles

- **Interchangeability**: Every model follows a strict interface defined in `interfaces.py`, allowing for seamless swapping during the "Arena" phase of Stage 4.
- **Context-Awareness**: Models are not treated as static entities but are selected and adjusted based on real-time market fingerprints.
- **Tensor Preparation**: Specialized preprocessing in `adapters/data_preparation.py` ensures data is correctly shaped (e.g., 3D tensors for LSTM) before reaching the models.

**Status**: Unified ML Architecture Active (Stage 4 & 5 Core)
**Primary Factory**: `src/models/factory.py`
