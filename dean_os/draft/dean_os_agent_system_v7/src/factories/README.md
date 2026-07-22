# Factories Module (`src/factories/`)

The **Factories Module** serves as the central **Model Lifecycle Manager** for the entire trading system. It provides a unified and standardized way to instantiate, configure, and manage machine learning models, ensuring that the pipeline remains decoupled from specific model implementations.

### Core Components

1.  **`model_factory.py`**: A thread-safe **Singleton** class responsible for the dynamic discovery and instantiation of models. It automatically scans the `src/models/` directory for any class that implements the `BaseModel` interface. This allows the system to support a "pluggable" architecture where adding a new model to the repository immediately makes it available to the training and prediction stages.

### The Model Contract

The strictly defined **contract** that every model in the system must follow is now located at `src/models/interfaces.py` (`BaseModel`). By enforcing a consistent API (requiring methods like `train()`, `predict()`, and `save_model()`), it guarantees that the `PipelineOrchestrator` can interact with any model—from a simple Linear Regression to a complex Transformer—without knowing its internal complexity.

### Role in the Pipeline

This factory acts as the **'Production Line'** of the project. It bridges the gap between the **Model Repository** (where architectures are defined) and the **Training/Prediction Stages**. 

Key advantages of this approach:
- **Seamless Scalability**: You can add new models to the 'Unified Model Repository' (`src/models/`) without changing a single line of code in the Pipeline or Training modules.
- **Unified Configuration**: The factory integrates with the `UnifiedConfigManager` to load model-specific parameters, allowing for centralized configuration management.
- **Dynamic Selection**: Supports an **Intelligent Model Selector** by providing a reliable way to spin up "Champion" models based on the current market context.

### Integration
- **Modeling Stage**: Used by the `ModelingStage` to create candidate models for evaluation.
- **Prediction Stage**: Used to load and instantiate trained models for generating real-time forecasts.