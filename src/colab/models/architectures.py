import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_sz):
        super().__init__()
        self.lstm = nn.LSTM(input_sz, 64, 2, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        out, _ = self.lstm(x.unsqueeze(1))
        return self.fc(out[:, -1, :])

class GRUModel(nn.Module):
    def __init__(self, input_sz):
        super().__init__()
        self.gru = nn.GRU(input_sz, 64, 2, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        out, _ = self.gru(x.unsqueeze(1))
        return self.fc(out[:, -1, :])

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

class TransformerModel(nn.Module):
    def __init__(self, input_sz):
        super().__init__()
        self.embedding = nn.Linear(input_sz, 64)
        encoder_layer = nn.TransformerEncoderLayer(64, 4, dim_feedforward=128, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, 2)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        x = self.embedding(x.unsqueeze(1))
        x = self.transformer(x)
        return self.fc(x[:, -1, :])

class AutoencoderModel(nn.Module):
    def __init__(self, input_sz):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_sz, 64), nn.ReLU(), nn.Linear(64, 32))
        self.decoder = nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, x):
        encoded = self.encoder(x)
        return self.decoder(encoded)
