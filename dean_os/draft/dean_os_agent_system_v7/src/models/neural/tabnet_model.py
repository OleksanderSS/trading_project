# src/models/neural/tabnet_model.py

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# Warning: pytorch_tabnet is not a standard library.
# Ensure it is installed: pip install pytorch-tabnet
try:
    from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor
except ImportError:
    TabNetRegressor = None
    TabNetClassifier = None

from src.core.logging.logger import ProjectLogger
from src.models.interfaces import BaseModel

META_EXTENSION = '.meta'


class TabNetModel(BaseModel):
    """
    Wrapper for TabNet model matching BaseModel interface.
    """
    def __init__(self, task_type: str = "regression", **kwargs):
        if TabNetRegressor is None:
            raise ImportError("pytorch_tabnet not installed. Please install it to use TabNetModel.")

        super().__init__(model_type="tabnet", task_type=task_type)
        self.model = self._create_model_instance(**kwargs)
        self.logger = ProjectLogger.get_logger(self.__class__.__name__)

    def _create_model_instance(self, **kwargs):
        """Creates TabNet model instance based on task type."""
        if self.task_type == "classification":
            return TabNetClassifier(**kwargs)
        else:
            return TabNetRegressor(**kwargs)

    def train(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> dict:
        """
        Trains model TabNet.

        :param X: Training data (features).
        :param y: Навчальні дані (Target variable..
        :param kwargs: Додаткові параметри для методу `fit`, наприклад, `eval_set`.
        """
        self.feature_cols = X.columns.tolist()

        # TabNet вимагає, щоб X та y були у форматі np.ndarray
        x_np = X.values.astype(np.float32)
        y_np = y.values.reshape(-1, 1).astype(np.float32)

        self.logger.info(f"Train the model {self.name}...")

        # Використовуємо параметри з kwargs, якщо вони надані
        fit_params = {
            "max_epochs": 50,
            "patience": 10,
            "batch_size": 256,
            **kwargs
        }

        self.model.fit(
            X_train=x_np,
            y_train=y_np,
            **fit_params
        )
        self.is_trained = True
        self.logger.info("Training завершено.")
        return {"status": "success", "message": f"Модель {self.name} успішно натренована."}

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Makes predictions.за допомогою натренованої моделі.

        :param X: Дані для Prediction.
        :return: Масив прогнозів.
        """
        if not self.is_trained:
            raise RuntimeError("Модель ще не натренована. Викличте метод `train` перед Predictionм.")

        x_np = X[self.feature_cols].values
        self.logger.info(f"Створення прогнозів з моделлю {self.name}...")
        predictions = self.model.predict(x_np)
        return predictions.flatten()

    def save_model(self, path: str) -> bool:
        """
        Зберігає натреновану модель TabNet.

        :param path: Шлях для збереження моделі (файл .zip).
        :return: True, якщо збереження пройшло успішно.
        """
        if not self.is_trained:
            self.logger.warning("Спроба зберегти ненатреновану модель.")
            return False

        # Зберігаємо саму модель TabNet
        model_path = self.model.save_model(path) # Повертає шлях до .zip

        # Зберігаємо метадані (наприклад, список ознак)
        metadata = {
            'feature_cols': self.feature_cols,
            'task_type': self.task_type
        }
        # Шлях до файлу метаданих буде поруч з моделлю
        metadata_path = Path(model_path).with_suffix(META_EXTENSION)
        joblib.dump(metadata, metadata_path)

        self.logger.info(f"Модель збережено в {model_path} та метадані в {metadata_path}")
        return True

    def load_model(self, path: str) -> bool:
        """
        Завантажує натреновану модель TabNet.

        :param path: Шлях до збереженої моделі (файл .zip).
        :return: True, якщо завантаження пройшло успішно.
        """
        try:
            # Завантажуємо метадані
            model_path = self._resolve_model_artifact_path(
                path,
                allowed_suffixes={'.zip'},
            )
            metadata_path = self._resolve_model_artifact_path(
                Path(path).with_suffix(META_EXTENSION),
                allowed_suffixes={META_EXTENSION},
            )
            metadata = joblib.load(metadata_path)  # audit-ignore: UNSAFE_MODEL_OR_PICKLE_LOAD
            self.feature_cols = metadata['feature_cols']
            self.task_type = metadata['task_type']

            # Створюємо екземпляр моделі і завантажуємо стан
            self.model = self._create_model_instance()
            self.model.load_model(str(model_path))

            self.is_trained = True
            self.logger.info(f"Модель успішно завантажено з {path}")
            return True
        except FileNotFoundError:
            self.logger.error(f"Файл моделі або метаданих не знайдено за шляхом: {path}")
            return False
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.logger.error(f"Помилка під час завантаження моделі: {e}")
            return False
