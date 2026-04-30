"""
Фабрика для створення моделей різних типів
"""
from typing import Optional, Any
import warnings
warnings.filterwarnings('ignore')


class ModelFactory:
    """Фабрика для створення моделей"""

    @staticmethod
    def create_model(model_type: str, input_size: int) -> Any:
        """Створити модель за типом"""
        try:
            import torch
            if torch is not None:
                return ModelFactory._create_torch_model(model_type, input_size)
        except ImportError:
            pass

        return ModelFactory._create_sklearn_fallback_model(model_type, input_size)

    @staticmethod
    def _create_torch_model(model_type: str, input_size: int) -> Any:
        """Створити PyTorch модель"""
        import torch
        import torch.nn as nn

        model_creators = {
            'mlp': ModelFactory._create_mlp_model,
            'lstm': ModelFactory._create_lstm_model,
            'gru': ModelFactory._create_gru_model,
            'cnn': ModelFactory._create_cnn_model,
            'transformer': ModelFactory._create_transformer_model,
            'tabnet': ModelFactory._create_tabnet_model,
            'autoencoder': ModelFactory._create_autoencoder_model,
        }

        creator = model_creators.get(model_type.lower())
        if creator:
            return creator(input_size)
        
        return ModelFactory._create_mlp_model(input_size)

    @staticmethod
    def _create_sklearn_fallback_model(model_type: str, input_size: int) -> Any:
        """Створити sklearn модель як fallback"""
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.neural_network import MLPRegressor

        if model_type.lower() in ['lstm', 'gru', 'cnn', 'transformer']:
            return MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=1000, random_state=42)
        
        return RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            min_samples_leaf=4,
            max_features='sqrt'
        )

    @staticmethod
    def _create_mlp_model(input_size: int) -> Any:
        """Створити MLP модель"""
        import torch.nn as nn
        
        return nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    @staticmethod
    def _create_lstm_model(input_size: int) -> Any:
        """Створити LSTM модель"""
        import torch.nn as nn
        
        class LSTMModel(nn.Module):
            def __init__(self, input_size):
                super().__init__()
                self.lstm = nn.LSTM(input_size, 128, batch_first=True)
                self.fc = nn.Linear(128, 1)

            def forward(self, x):
                lstm_out, _ = self.lstm(x)
                return self.fc(lstm_out[:, -1, :])

        return LSTMModel(input_size)

    @staticmethod
    def _create_gru_model(input_size: int) -> Any:
        """Створити GRU модель"""
        import torch.nn as nn
        
        class GRUModel(nn.Module):
            def __init__(self, input_size):
                super().__init__()
                self.gru = nn.GRU(input_size, 128, batch_first=True)
                self.fc = nn.Linear(128, 1)

            def forward(self, x):
                gru_out, _ = self.gru(x)
                return self.fc(gru_out[:, -1, :])

        return GRUModel(input_size)

    @staticmethod
    def _create_cnn_model(input_size: int) -> Any:
        """Створити CNN модель"""
        import torch.nn as nn
        
        class CNNModel(nn.Module):
            def __init__(self, input_size):
                super().__init__()
                self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
                self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
                self.fc = nn.Linear(64 * input_size, 1)

            def forward(self, x):
                x = x.unsqueeze(1)
                x = nn.functional.relu(self.conv1(x))
                x = nn.functional.relu(self.conv2(x))
                x = x.view(x.size(0), -1)
                return self.fc(x)

        return CNNModel(input_size)

    @staticmethod
    def _create_transformer_model(input_size: int) -> Any:
        """Створити Transformer модель"""
        import torch.nn as nn
        
        class TransformerModel(nn.Module):
            def __init__(self, input_size):
                super().__init__()
                self.embedding = nn.Linear(input_size, 64)
                encoder_layer = nn.TransformerEncoderLayer(d_model=64, nhead=4, batch_first=True)
                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
                self.fc = nn.Linear(64, 1)

            def forward(self, x):
                x = self.embedding(x.unsqueeze(1))
                x = self.transformer(x)
                return self.fc(x[:, -1, :])

        return TransformerModel(input_size)

    @staticmethod
    def _create_tabnet_model(input_size: int) -> Any:
        """Створити TabNet модель"""
        try:
            from pytorch_tabnet.tab_model import TabNetRegressor
            return TabNetRegressor()
        except ImportError:
            return ModelFactory._create_mlp_model(input_size)

    @staticmethod
    def _create_autoencoder_model(input_size: int) -> Any:
        """Створити Autoencoder модель"""
        import torch.nn as nn
        
        class AutoencoderModel(nn.Module):
            def __init__(self, input_size):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(input_size, 128),
                    nn.ReLU(),
                    nn.Linear(128, 64)
                )
                self.decoder = nn.Sequential(
                    nn.Linear(64, 128),
                    nn.ReLU(),
                    nn.Linear(128, 1)
                )

            def forward(self, x):
                encoded = self.encoder(x)
                return self.decoder(encoded)

        return AutoencoderModel(input_size)
