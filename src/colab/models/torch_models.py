"""PyTorch model architectures"""

import torch
import torch.nn as nn


def create_torch_model(model_type, input_size):
    """Create PyTorch model"""
    
    model_creators = {
        'mlp': create_mlp_model,
        'lstm': create_lstm_model,
        'gru': create_gru_model,
        'cnn': create_cnn_model,
        'transformer': create_transformer_model,
        'tabnet': create_tabnet_model,
        'random_forest': create_random_forest_wrapper,
        'autoencoder': create_autoencoder_model
    }
    
    creator = model_creators.get(model_type)
    if creator is None:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return creator(input_size)


def create_mlp_model(input_size):
    """Create MLP model"""
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


def create_lstm_model(input_size):
    """Create LSTM model"""
    
    class LSTMModel(nn.Module):
        def __init__(self, input_sz):
            super().__init__()
            self.lstm = nn.LSTM(
                input_sz, 64, 2, batch_first=True, dropout=0.2)
            self.fc = nn.Linear(64, 1)

        def forward(self, x):
            out, _ = self.lstm(x.unsqueeze(1))
            return self.fc(out[:, -1, :])
    
    return LSTMModel(input_size)


def create_gru_model(input_size):
    """Create GRU model"""
    
    class GRUModel(nn.Module):
        def __init__(self, input_sz):
            super().__init__()
            self.gru = nn.GRU(
                input_sz, 64, 2, batch_first=True, dropout=0.2)
            self.fc = nn.Linear(64, 1)

        def forward(self, x):
            out, _ = self.gru(x.unsqueeze(1))
            return self.fc(out[:, -1, :])
    
    return GRUModel(input_size)


def create_cnn_model(input_size):
    """Create CNN model"""
    
    class CNNModel(nn.Module):
        def __init__(self, input_sz):
            super().__init__()
            self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
            self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.fc = nn.Linear(64, 1)

        def forward(self, x):
            x = x.unsqueeze(1)
            x = torch.relu(self.conv1(x))
            x = torch.relu(self.conv2(x))
            return self.fc(self.pool(x).squeeze(-1))
    
    return CNNModel(input_size)


def create_transformer_model(input_size):
    """Create Transformer model"""
    
    class TransformerModel(nn.Module):
        def __init__(self, input_sz):
            super().__init__()
            self.embedding = nn.Linear(input_sz, 64)
            encoder_layer = nn.TransformerEncoderLayer(
                64, 4, dim_feedforward=128, dropout=0.2, batch_first=True)
            self.transformer = nn.TransformerEncoder(encoder_layer, 2)
            self.fc = nn.Linear(64, 1)

        def forward(self, x):
            x = self.embedding(x.unsqueeze(1))
            x = self.transformer(x)
            return self.fc(x[:, -1, :])
    
    return TransformerModel(input_size)


def create_tabnet_model(input_size):
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


def create_autoencoder_model(input_size):
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
