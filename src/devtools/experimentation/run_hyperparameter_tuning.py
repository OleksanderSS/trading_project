# src/devtools/experimentation/run_hyperparameter_tuning.py

from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor

from src.core.logging.logger import ProjectLogger
from src.scripts.optimization.factory import OptimizationFactory


def main():
    """
    Demonstrates the canonical use of the BayesianOptimizer for hyperparameter tuning.
    This script serves as a template for conducting ML experimentation and finding the
    optimal parameters for a given model and dataset.
    """
    logger = ProjectLogger.get_logger("HyperparameterTuningRun")
    logger.info("--- Starting Hyperparameter Tuning Demonstration ---")

    # 1. Generate a synthetic dataset for demonstration purposes
    X, y = make_regression(n_samples=1000, n_features=20, n_informative=15, noise=0.1, random_state=42)
    logger.info(f"Generated synthetic dataset with shape X: {X.shape}, y: {y.shape}")

    # 2. Define the hyperparameter search space for the model
    # This dictionary defines the parameters to tune, their type (int, float, categorical),
    # and their respective ranges or values.
    param_space = {
        "n_estimators": ["int", 100, 1000],
        "max_depth": ["int", 5, 50],
        "min_samples_split": ["int", 2, 20],
        "min_samples_leaf": ["int", 1, 10],
        "max_features": ["float", 0.1, 1.0]
    }
    logger.info(f"Defined parameter search space: {param_space}")

    # 3. Use the OptimizationFactory to get a Bayesian optimizer
    # We configure it with the model constructor, the search space, and the number of trials.
    try:
        optimizer = OptimizationFactory.get_optimizer(
            optimizer_type='bayesian',
            model_func=RandomForestRegressor, # The model class to be tuned
            param_space=param_space,
            n_trials=25  # Number of optimization trials (use more for real tasks)
        )
    except ImportError as e:
        logger.error(f"Failed to create optimizer: {e}")
        logger.error("Please ensure all required dependencies are installed.")
        return

    logger.info("Successfully created BayesianOptimizer.")

    # 4. Run the optimization process
    # The optimizer will now use Optuna to search for the best combination of parameters
    # that maximizes the objective function (in this case, negative mean squared error).
    logger.info("Starting optimization...")
    try:
        best_params = optimizer.optimize(X, y)

        if not best_params:
            logger.warning("Optimization did not yield any parameters. Check logs for errors.")
            return

        logger.info("--- Hyperparameter Tuning Finished ---")
        logger.info(f"Best Score (neg_mean_squared_error): {optimizer.best_score:.4f}")
        logger.info(f"Best Parameters Found: {best_params}")

        # 5. (Optional) Train the final model with the best parameters
        logger.info("Training final model with best parameters...")
        final_model = RandomForestRegressor(**best_params, random_state=42)
        final_model.fit(X, y)
        logger.info("Final model trained successfully.")

    except Exception as e:
        logger.critical(f"The optimization process failed critically: {e}", exc_info=True)

if __name__ == "__main__":
    main()
