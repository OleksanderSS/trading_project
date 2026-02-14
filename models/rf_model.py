# models/rf_model.py

from models.tree_models import train_rf_regressor, train_rf_classifier

def train_rf_model(X, y, task="regression"):
    """Wrapper for Random Forest моwhereлand"""
    if task == "classification":
        return train_rf_classifier(X, y)
    else:
        return train_rf_regressor(X, y)