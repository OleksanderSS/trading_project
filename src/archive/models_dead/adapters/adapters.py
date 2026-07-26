# src/models/adapters/adapters.py - Адаптери для різних середовищ виконання моделей

import logging
from typing import Any

import numpy as np
import pandas as pd

# Імпортуємо базовий інтерфейс
from ..interfaces import BaseModel

logger = logging.getLogger(__name__)

class LightModelInterface(BaseModel):
    """Інтерфейс для "легких" моделей, що тренуються локально."""

    def __init__(self, model_type: str, task_type: str = "regression"):
        super().__init__(model_type, task_type)
        # Виправлено імпорт
        from src.training.light_model_trainer import LightModelTrainer
        self.trainer = LightModelTrainer()

    def train(self, X: np.ndarray, y: np.ndarray, ticker: str = "DEFAULT", timeframe: str = "1d") -> dict[str, Any]:
        """Тренує "легку" модель за допомогою LightModelTrainer."""
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
        """Робить прогнози, використовуючи навчену модель."""
        if not self.is_trained or not hasattr(self, 'model_key'):
            raise ValueError("Модель ще не навчена або відсутній model_key.")

        if not self.feature_cols:
             self.feature_cols = [f"feature_{i}" for i in range(X.shape[1])]

        df = pd.DataFrame(X, columns=self.feature_cols)
        return self.trainer.predict(self.model_key, df)

    def save_model(self, path: str) -> bool:
        """Зберігає метадані моделі."""
        try:
            import joblib
            model_data = self.get_model_info()
            model_data['model_key'] = getattr(self, 'model_key', None)
            joblib.dump(model_data, path)
            return True
        except Exception as e:
            logger.error(f"Помилка збереження стану інтерфейсу легкої моделі: {e}", exc_info=True)
            return False

    def load_model(self, path: str) -> bool:
        """Завантажує метадані моделі."""
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
            logger.error(f"Помилка завантаження стану інтерфейсу легкої моделі: {e}", exc_info=True)
            return False

class HeavyModelInterface(BaseModel):
    """Інтерфейс для "важких" моделей, що делегує операції ColabManager."""

    def __init__(self, model_type: str, task_type: str = "regression"):
        super().__init__(model_type, task_type)
        self.colab_manager = None

    def _initialize_manager(self):
        if self.colab_manager is None:
            try:
                from utils.colab_manager import ColabManager
                self.colab_manager = ColabManager()
            except ImportError:
                logger.error("ColabManager не може бути імпортований. Операції з важкими моделями не будуть працювати.")
                raise

    def train(self, X: np.ndarray, y: np.ndarray, ticker: str = "DEFAULT", timeframe: str = "1d") -> dict[str, Any]:
        """Надсилає дані в Colab для тренування важкої моделі."""
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
        """Запитує прогнози у навченої важкої моделі в Colab."""
        if not self.is_trained or not hasattr(self, 'model_id'):
            raise ValueError("Важка модель не навчена або відсутній model_id.")
        self._initialize_manager()

        data_payload = {"X": X.tolist(), "model_id": self.model_id}
        result = self.colab_manager.predict_heavy_model(data_payload)

        if result.get("success"):
            return np.array(result["predictions"])
        else:
            error_msg = result.get('error', 'Невідома помилка прогнозування')
            raise ValueError(f"Помилка прогнозування у важкій моделі: {error_msg}")

    def save_model(self, path: str) -> bool:
        """Ініціює операцію збереження в Colab."""
        if not self.colab_manager or not hasattr(self, 'model_id'):
            logger.warning("Неможливо зберегти модель: Colab менеджер або model_id недоступні.")
            return False

        result = self.colab_manager.save_heavy_model(self.model_id, path)
        return result.get("success", False)

    def load_model(self, path: str) -> bool:
        """Ініціює операцію завантаження в Colab."""
        self._initialize_manager()
        result = self.colab_manager.load_heavy_model(path)

        if result.get("success"):
            self.is_trained = True
            self.model_id = result.get("model_id")
            self.metrics = result.get("metrics", {})
            return True
        return False
