# models/gru_model.py

import torch
import torch.nn as nn
import numpy as np
from utils.logger import ProjectLogger

logger = ProjectLogger.get_logger(__name__)

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=1, dropout=0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.gru = nn.GRU(input_size, hidden_size, num_layers, 
                         batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        out, _ = self.gru(x)
        out = self.dropout(out[:, -1, :])  # Осandннandй timestep
        out = self.fc(out)
        return out

def train_gru_model(X, y, classification=False):
    """GRU model for часових рядandв"""
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if X.ndim == 2:
            # Перетворюємо 2D в 3D for GRU
            seq_len = min(10, X.shape[0] // 2)
            X_seq = np.array([X[i:i+seq_len] for i in range(X.shape[0]-seq_len+1)])
            y_seq = y[seq_len-1:]
        else:
            X_seq, y_seq = X, y
            
        model = GRUModel(X_seq.shape[2], hidden_size=64, num_layers=2)
        model.to(device)
        
        criterion = nn.MSELoss() if not classification else nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Тренування
        model.train()
        X_tensor = torch.FloatTensor(X_seq).to(device)
        y_tensor = torch.FloatTensor(y_seq).to(device)
        
        for epoch in range(50):
            optimizer.zero_grad()
            outputs = model(X_tensor)
            loss = criterion(outputs.squeeze(), y_tensor)
            loss.backward()
            optimizer.step()
            
        logger.info("[OK] GRU model натренована")
        return model
        
    except Exception as e:
        logger.error(f"[ERROR] Error GRU: {e}")
        return None