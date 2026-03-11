# 🧪 ML Experimentation Toolkit

This directory contains tools and scripts designed for machine learning experimentation, prototyping, and performance analysis. It serves as a sandbox for ML engineers to test new models, tune hyperparameters, and validate hypotheses before integrating them into the main production pipeline.

## Key Scripts and Workflows

### Hyperparameter Tuning (`run_hyperparameter_tuning.py`)

*   **Purpose**: This script provides a canonical example of how to perform hyperparameter optimization for any scikit-learn compatible model using the project's `BayesianOptimizer`.
*   **How it Works**:
    1.  It defines a hyperparameter search space (`param_space`).
    2.  It uses the `OptimizationFactory` to instantiate the `BayesianOptimizer`.
    3.  The optimizer leverages `Optuna` to intelligently search the parameter space and find the combination that maximizes the model's performance (evaluated via cross-validation).
*   **Usage**: This script is the recommended starting point for tuning new models. Copy and adapt it for your specific model and dataset.

    ```bash
    python src/devtools/experimentation/run_hyperparameter_tuning.py
    ```

### Future Experiments

This directory can be extended with other experimental scripts, such as:

*   **`feature_importance_analysis.py`**: For analyzing the predictive power of different feature sets.
*   **`model_comparison.py`**: For benchmarking the performance of different model architectures on a fixed dataset.
*   **`alternative_data_validation.py`**: For testing the value of new, alternative data sources.
