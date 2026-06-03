"""Sklearn fallback models when torch is not available"""


from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger(__name__)


def create_sklearn_fallback_model(model_type, input_size):
    """Create sklearn fallback model when torch is not available"""
    logger.info(f"   ⚠️ torch не доступний, використовуємо sklearn fallback для {model_type}")
    
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    
    class SklearnModelWrapper:
        def __init__(self, model_type, input_size):
            if model_type in ['mlp', 'cnn', 'lstm', 'gru', 'transformer']:
                self.model = RandomForestRegressor(
                    n_estimators=50,
                    max_depth=8,
                    min_samples_leaf=1,
                    max_features='sqrt',
                    random_state=42
                )
            else:
                self.model = LinearRegression()
            self.input_size = input_size
            
        def __call__(self, x):
            return self.forward(x)
            
        def forward(self, x):
            if hasattr(x, 'detach'):
                x_np = x.detach().cpu().numpy()
            else:
                x_np = x
            import numpy as np
            result = self.model.predict(x_np)
            return _create_fake_tensor(result)
                
        def parameters(self):
            return []
                
        def train(self):
            # Set model to training mode (no-op for sklearn wrapper)
            pass
                
        def eval(self):
            # Set model to evaluation mode (no-op for sklearn wrapper)
            pass
                
        def state_dict(self):
            return {'model': self.model}
                
    return SklearnModelWrapper(model_type, input_size)


def _create_fake_tensor(data):
    """Create fake tensor object for sklearn compatibility"""
    class FakeTensor:
        def __init__(self, data):
            self.data = data
        def numpy(self):
            return self.data
        def flatten(self):
            return self.data.flatten()
    return FakeTensor(data)
