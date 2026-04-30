# src/models/adapters/adapters.py - Adapters for light/heavy model integration

import pandas as pd
import numpy as np
from typing import Dict, Any

# Model interfaces
from ..interfaces import BaseModel
from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger(__name__)


class LightModelInterface(BaseModel):
    """Light interface for local model training and inference."""

    def __init__(self, model_type: str, task_type: str = "regression"):
        super().__init__(model_type, task_type)
        # Initialize trainer
        from src.training.light_model_trainer import LightModelTrainer
        self.trainer = LightModelTrainer()

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        ticker: str = "DEFAULT",
        timeframe: str = "1d"
    ) -> Dict[str, Any]:
        """Train light model using LightModelTrainer."""
        feature_cols = [f"feature_{i}" for i in range(X.shape[1])]
        df = pd.DataFrame(X, columns=feature_cols)
        df["target"] = y

        result = self.trainer.train_light_model(
            features_df=df,
            model_type=self.model_type,
            ticker=ticker,
            timeframe=timeframe,
            target_col="target",
            task_type=self.task_type
        )

        if result.get("status") == "success":
            self.is_trained = True
            self.feature_cols = feature_cols
            self.metrics = result.get("metrics", {})
            self.model_key = result.get("model_key")

        return result

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Makes predictions using trained model."""
        if not self.is_trained or not hasattr(self, 'model_key'):
            raise ValueError(
                "Model not trained or missing model_key."
            )

        if not self.feature_cols:
            self.feature_cols = [f"feature_{i}" for i in range(X.shape[1])]

        df = pd.DataFrame(X, columns=self.feature_cols)
        return self.trainer.predict(self.model_key, df)

    def save_model(self, path: str) -> bool:
        """Save model metadata."""
        try:
            import joblib
            model_data = self.get_model_info()
            model_data['model_key'] = getattr(self, 'model_key', None)
            joblib.dump(model_data, path)
            return True
        except Exception as e:
            logger.error(
                f"Error saving model metadata: {e}",
                exc_info=True
            )
            return False

    def load_model(self, path: str) -> bool:
        """Load model metadata."""
        try:
            import joblib
            model_data = joblib.load(path)

            self.model_type = model_data["model_type"]
            self.task_type = model_data["task_type"]
            self.is_trained = model_data["is_trained"]
            self.feature_cols = model_data["feature_cols"]
            self.metrics = model_data["metrics"]
            self.model_key = model_data.get("model_key")

            from src.training.light_model_trainer import LightModelTrainer
            self.trainer = LightModelTrainer()

            return self.is_trained
        except Exception as e:
            logger.error(
                f"Error loading model metadata: {e}",
                exc_info=True
            )
            return False


class HeavyModelInterface(BaseModel):
    """Heavy interface for Colab-based model training."""

    def __init__(self, model_type: str, task_type: str = "regression"):
        super().__init__(model_type, task_type)
        self.colab_manager = None

    def _initialize_manager(self):
        if self.colab_manager is None:
            try:
                from utils.colab_manager import ColabManager
                self.colab_manager = ColabManager()
            except ImportError:
                logger.error(
                    "ColabManager not available. "
                    "Install required dependencies."
                )
                raise

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        ticker: str = "DEFAULT",
        timeframe: str = "1d"
    ) -> Dict[str, Any]:
        """Train model via Colab training pipeline."""
        self._initialize_manager()

        data_payload = {
            "X": X.tolist(),
            "y": y.tolist(),
            "model_type": self.model_type,
            "task_type": self.task_type,
            "ticker": ticker,
            "timeframe": timeframe
        }

        result = self.colab_manager.train_heavy_model(data_payload)

        if result.get("success"):
            self.is_trained = True
            self.metrics = result.get("metrics", {})
            self.model_id = result.get("model_id")

        return result

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Get predictions from heavy model via Colab."""
        if not self.is_trained or not hasattr(self, 'model_id'):
            raise ValueError(
                "Heavy model not trained or missing model_id."
            )
        self._initialize_manager()

        data_payload = {"X": X.tolist(), "model_id": self.model_id}
        result = self.colab_manager.predict_heavy_model(data_payload)

        if result.get("success"):
            return np.array(result["predictions"])
        else:
            error_msg = result.get('error', 'Prediction failed')
            raise ValueError(
                f"Prediction failed for heavy model: {error_msg}"
            )

    def save_model(self, path: str) -> bool:
        """Save model metadata via Colab."""
        if not self.colab_manager or not hasattr(self, 'model_id'):
            logger.warning(
                "Cannot save model: Colab manager or model_id missing."
            )
            return False

        result = self.colab_manager.save_heavy_model(self.model_id, path)
        return result.get("success", False)

    def load_model(self, path: str) -> bool:
        """Load model metadata via Colab."""
        self._initialize_manager()
        result = self.colab_manager.load_heavy_model(path)

        if result.get("success"):
            self.is_trained = True
            self.model_id = result.get("model_id")
            self.metrics = result.get("metrics", {})
            return True
        return False
