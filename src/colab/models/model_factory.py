"""Factory for creating different model architectures"""

from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger(__name__)

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    nn = None
    TORCH_AVAILABLE = False


def create_model(model_type, input_size):
    """Create model based on type with fallback to sklearn if torch unavailable"""
    if not TORCH_AVAILABLE:
        return _create_sklearn_fallback_model(model_type, input_size)
    
    return _create_torch_model(model_type, input_size)


def _create_sklearn_fallback_model(model_type, input_size):
    """Create sklearn fallback model when torch is not available"""
    logger.warning(f"   ⚠️ torch не доступний, використовуємо sklearn fallback для {model_type}")
    
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


def _create_torch_model(model_type, input_size):
    """Create PyTorch model"""
    import torch
    import torch.nn as nn

    # audit-ignore: AUTOENCODER_ROUTING_REVIEW
    from src.colab.models.architectures import LSTMModel, GRUModel, CNNModel, TransformerModel, AutoencoderModel

    model_creators = {
        'mlp': _create_mlp_model,
        'lstm': lambda sz: LSTMModel(sz),
        'gru': lambda sz: GRUModel(sz),
        'cnn': lambda sz: CNNModel(sz),
        'transformer': lambda sz: TransformerModel(sz),
        'tabnet': _create_tabnet_model,
        'random_forest': _create_random_forest_wrapper,
        'autoencoder': lambda sz: AutoencoderModel(sz)
    }
    
    creator = model_creators.get(model_type)
    if creator is None:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return creator(input_size)


def _create_mlp_model(input_size):
    """Create MLP model"""
    import torch.nn as nn
    return nn.Sequential(
        nn.Linear(input_size, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 1)
    )


def _create_lstm_model(input_size):
    """Create LSTM model"""
    # Deprecated in favor of direct creator lambda in _create_torch_model
    pass


def _create_gru_model(input_size):
    """Create GRU model"""
    pass


def _create_cnn_model(input_size):
    """Create CNN model"""
    pass


def _create_transformer_model(input_size):
    """Create Transformer model"""
    pass


def _create_tabnet_model(input_size):
    """Create TabNet model (fallback to MLP)"""
    return nn.Sequential(
        nn.Linear(input_size, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(64, 1)
    )


def _create_random_forest_wrapper(input_size):
    """Create RandomForest wrapper for torch compatibility"""
    from sklearn.ensemble import RandomForestRegressor
    
    class RandomForestWrapper:
        def __init__(self, input_size):
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                random_state=42
            )
            self.input_size = input_size
            
        def __call__(self, x):
            return self.forward(x)
            
        def forward(self, x):
            if hasattr(x, 'detach'):
                x_np = x.detach().cpu().numpy()
            else:
                x_np = x
            return torch.tensor(self.model.predict(x_np), dtype=torch.float32)
                
        def parameters(self):
            return []
                
        def train(self):
            # Set model to training mode (no-op for sklearn wrapper)
            self.model.train = True
                
        def eval(self):
            # Set model to evaluation mode (no-op for sklearn wrapper)
            self.model.train = False
    
    return RandomForestWrapper(input_size)


def _create_autoencoder_model(input_size):
    """Create Autoencoder model"""
    class AutoencoderModel(nn.Module):
        def __init__(self, input_sz):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_sz, 64), nn.ReLU(),
                nn.Linear(64, 32), nn.ReLU()
            )
            self.decoder = nn.Sequential(
                nn.Linear(32, 16), nn.ReLU(),
                nn.Linear(16, 1)
            )

        def forward(self, x):
            return self.decoder(self.encoder(x))
    
    return AutoencoderModel(input_size)
