from typing import Dict, Any, Optional
from pathlib import Path
import numpy as np

class OverfittingConfig:
    """Configuration and thresholds for the Overfitting Detector."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

        # Overfitting signal types
        self.OVERFITTING_SIGNALS = {
            'train_val_gap': {
                'description': 'Large train-validation performance gap',
                'threshold': 0.1,
                'severity': 'high'
            },
            'learning_curve': {
                'description': 'Unstable learning curve patterns',
                'threshold': 0.15,
                'severity': 'medium'
            },
            'cv_variance': {
                'description': 'High cross-validation variance',
                'threshold': 0.05,
                'severity': 'medium'
            },
            'complexity_penalty': {
                'description': 'Model complexity vs performance trade-off',
                'threshold': 0.02,
                'severity': 'low'
            }
        }
        
        self.thresholds = self.OVERFITTING_SIGNALS.copy()
        self.thresholds.update(self.config.get('thresholds', {}))
        
        self.cv_folds = self.config.get('cv_folds', 5)
        self.scoring_metric = self.config.get('scoring_metric', 'neg_mean_squared_error')
        self.train_sizes = self.config.get('train_sizes', np.linspace(0.1, 1.0, 10))
        self.enable_visualization = self.config.get('enable_visualization', True)
        self.save_plots = self.config.get('save_plots', True)
        self.storage_path = Path(self.config.get('storage_path', 'data/analysis/overfitting'))
