# models/xgb_model.py

from models.tree_models import train_xgb_regressor, train_xgb_classifier

def train_xgb_model(X, y, task="regression"):
    """Wrapper for XGBoost моwhereлand"""
    if task == "classification":
        return train_xgb_classifier(X, y)
    else:
        return train_xgb_regressor(X, y)