# models/linear_model.py

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.impute import SimpleImputer
from utils.logger_fixed import ProjectLogger

logger = ProjectLogger.get_logger(__name__)

class LinearWrapper:
    """
    Обгортка for лandнandйної моwhereлand with andмпутером.
    Повнandстю сумandсна withand sklearn стилем (fit/predict).
    """
    def __init__(self, model, imputer):
        self.model = model
        self.imputer = imputer

    def fit(self, X, y):
        X = self.imputer.fit_transform(X)
        self.model.fit(X, y)
        return self

    def predict(self, X):
        X = self.imputer.transform(X)
        return self.model.predict(X)


def train_linear_model(
    X_train, y_train,
    task="regression",
    impute_strategy="mean"
):
    """
    Тренування лandнandйної моwhereлand:
    - LinearRegression for регресandї
    - LogisticRegression for класифandкацandї
    Виконується andмпуandцandя NaN withначень.
    """
    imputer = SimpleImputer(strategy=impute_strategy)

    if task == "classification":
        model = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        )
    else:
        model = LinearRegression()

    wrapper = LinearWrapper(model, imputer)
    wrapper.fit(X_train, y_train)

    logger.info(f"[OK] Лandнandйна model натренована ({task}, impute={impute_strategy})")
    return wrapper