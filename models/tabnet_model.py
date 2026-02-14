# models/tabnet_model.py

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
from utils.logger import ProjectLogger

logger = ProjectLogger.get_logger(__name__)

def train_tabnet_model(X, y, task="regression"):
    """TabNet model for andбличних data"""
    try:
        from pytorch_tabnet.tab_model import TabNetRegressor, TabNetClassifier
        
        if task == "classification":
            model = TabNetClassifier(verbose=0, seed=42)
        else:
            model = TabNetRegressor(verbose=0, seed=42)
        
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            max_epochs=50,
            patience=10,
            batch_size=256,
            virtual_batch_size=128
        )
        
        logger.info("[OK] TabNet model натренована")
        return model
        
    except ImportError:
        logger.warning("[WARN] pytorch-tabnet not всandновлено, пропускаю TabNet")
        return None
    except Exception as e:
        logger.error(f"[ERROR] Error TabNet: {e}")
        return None