"""PyTorch model architectures"""


def create_torch_model(model_type, input_size):
    """Create PyTorch model"""
    import torch.nn as nn

    # audit-ignore: AUTOENCODER_ROUTING_REVIEW
    from src.colab.models.architectures import (  # audit-ignore: AUTOENCODER_ROUTING_REVIEW
        AutoencoderModel,
        CNNModel,
        GRUModel,
        LSTMModel,
        TransformerModel,
    )

    model_creators = {
        'mlp': lambda sz: nn.Sequential(
            nn.Linear(sz, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        ),
        'lstm': lambda sz: LSTMModel(sz),
        'gru': lambda sz: GRUModel(sz),
        'cnn': lambda sz: CNNModel(sz),
        'transformer': lambda sz: TransformerModel(sz),
        'tabnet': lambda sz: nn.Sequential(
            nn.Linear(sz, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        ),
        'random_forest': create_random_forest_wrapper,
        'autoencoder': lambda sz: AutoencoderModel(sz)  # audit-ignore: AUTOENCODER_ROUTING_REVIEW
    }

    creator = model_creators.get(model_type)
    if creator is None:
        raise ValueError(f"Unknown model type: {model_type}")

    return creator(input_size)


def create_mlp_model(input_size):
    """Create MLP model"""
    # Deprecated
    pass


def create_lstm_model(input_size):
    """Create LSTM model"""
    from src.colab.models.architectures import LSTMModel
    return LSTMModel(input_size)


def create_gru_model(input_size):
    """Create GRU model"""
    from src.colab.models.architectures import GRUModel
    return GRUModel(input_size)


def create_cnn_model(input_size):
    """Create CNN model"""
    from src.colab.models.architectures import CNNModel
    return CNNModel(input_size)


def create_transformer_model(input_size):
    """Create Transformer model"""
    from src.colab.models.architectures import TransformerModel
    return TransformerModel(input_size)


def create_tabnet_model(input_size):
    """Create TabNet model (fallback to MLP)"""
    import torch.nn as nn
    return nn.Sequential(
        nn.Linear(input_size, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(64, 1)
    )


def create_random_forest_wrapper(input_size):
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
            import torch
            if hasattr(x, 'detach'):
                x_np = x.detach().cpu().numpy()
            else:
                x_np = x
            return torch.tensor(self.model.predict(x_np), dtype=torch.float32)

        def parameters(self):
            return []

        def train(self):
            self.model.train = True

        def eval(self):
            self.model.train = False

    return RandomForestWrapper(input_size)


def create_autoencoder_model(input_size):  # audit-ignore: AUTOENCODER_ROUTING_REVIEW
    """Create Autoencoder model"""
    from src.colab.models.architectures import AutoencoderModel  # audit-ignore: AUTOENCODER_ROUTING_REVIEW
    return AutoencoderModel(input_size)  # audit-ignore: AUTOENCODER_ROUTING_REVIEW
