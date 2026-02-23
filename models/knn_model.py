# models/knn_model.py

import joblib
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.impute import SimpleImputer
from utils.logger import ProjectLogger

logger = ProjectLogger.get_logger(__name__)

def train_knn_model(
    X_train, y_train,
    task: str = "regression",
    n_neighbors: int = 5,
    weights: str = "uniform",
    impute_strategy: str = "mean",
    save_path: str = None,
    **kwargs
):
    """
    Тренування KNN:
    - KNeighborsRegressor for регресandї
    - KNeighborsClassifier for класифandкацandї
    Виконується andмпуandцandя NaN withначень.
    """

    #  Імпуandцandя пропускandв у тренувальних data
    imputer = SimpleImputer(strategy=impute_strategy)
    X_train = imputer.fit_transform(X_train)

    #  Вибandр моwhereлand forлежно вandд forдачand
    if task == "classification":
        model = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights=weights,
            **kwargs
        )
    else:
        model = KNeighborsRegressor(
            n_neighbors=n_neighbors,
            weights=weights,
            **kwargs
        )

    #  Навчання моwhereлand
    model.fit(X_train, y_train)
    logger.info(
        f"[OK] KNN натренований ({task}, n_neighbors={n_neighbors}, weights={weights}, impute={impute_strategy})"
    )

    #  Збереження моwhereлand (опцandонально)
    if save_path:
        joblib.dump((model, imputer), save_path)
        logger.info(f" Моwhereль KNN withбережена у {save_path}")

    #  Обгортка for andмпуandцandї and на тестових data
    class KNNWrapper:
        def __init__(self, model, imputer):
            self.model = model
            self.imputer = imputer

        def predict(self, X):
            X = self.imputer.transform(X)
            return self.model.predict(X)

        def fit(self, X, y):
            X = self.imputer.fit_transform(X)
            return self.model.fit(X, y)

    return KNNWrapper(model, imputer)