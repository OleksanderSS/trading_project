# models/mlp_model.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from utils.logger_fixed import ProjectLogger

logger = ProjectLogger.get_logger(__name__)


class MLPModel(nn.Module):
    """
    Простий багатошаровий перцептрон (MLP) for forдач регресandї or класифandкацandї.
    Має метод .predict(), so that бути сумandсним withand Stage 4.
    """
    def __init__(self, input_size: int, hidden_size: int = 64,
                 output_size: int = 1, dropout: float = 0.2,
                 task: str = "regression"):
        super().__init__()
        if input_size <= 0:
            raise ValueError("input_size має бути > 0")

        self.task = task
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(hidden_size // 2, output_size)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.relu(self.fc2(x))
        x = self.dropout2(x)
        return self.fc3(x)

    def predict(self, X):
        """Метод for прогноwithування, сумandсний withand sklearnподandбними моwhereлями."""
        self.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(
                np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
            )
            device = next(self.parameters()).device
            X_tensor = X_tensor.to(device)
            raw_preds = self.forward(X_tensor).cpu().numpy().flatten()

        if self.task == "classification":
            probs = 1 / (1 + np.exp(-raw_preds))  # sigmoid
            return (probs >= 0.5).astype(int)
        return raw_preds

def train_mlp(X_train, y_train, input_size=None, epochs=50, lr=0.001,
              batch_size=32, validation_split=0.2, early_stopping=10,
              shuffle=True, task="regression"):
    """
    Тренування MLP with валandдацandєю and early stopping.
    """

    # --- Санandandрка NaN/inf ---
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    y_train = np.nan_to_num(y_train, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    
    # --- Перевandрка на валandднand данand ---
    if len(X_train) == 0 or len(y_train) == 0:
        raise ValueError("Порожнand данand for тренування")
    
    # --- Перевandрка на варandативнandсть y ---
    if np.all(y_train == y_train[0]):  # Всand y однаковand
        logger.warning("y_train має нульову варandативнandсть, використовуюмо випадковand прогноwithи")
        # Створюємо просту model що поверandє notвеликand випадковand values
        class DummyMLP:
            def predict(self, X):
                return np.random.normal(0, 0.001, len(X))
            def __call__(self, x):
                return self.predict(x)
        return DummyMLP(), 0.0

    # --- Динамandчний input_size ---
    if input_size is None:
        if X_train.ndim != 2 or X_train.shape[1] == 0:
            raise ValueError("X_train має бути 2D and мandстити хоча б одну оwithнаку")
        input_size = X_train.shape[1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLPModel(input_size, task=task).to(device)

    # Loss forлежно вandд forдачand
    criterion = nn.BCEWithLogitsLoss() if task == "classification" else nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # --- Split на train/val ---
    split_idx = max(1, int(len(X_train) * (1 - validation_split)))
    X_tr, X_val = X_train[:split_idx], X_train[split_idx:]
    y_tr, y_val = y_train[:split_idx], y_train[split_idx:]

    # --- TensorDataset / DataLoader ---
    train_dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_tr), torch.tensor(y_tr).reshape(-1, 1)
    )
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()

        # --- Validation ---
        model.eval()
        with torch.no_grad():
            Xv = torch.tensor(X_val, dtype=torch.float32).to(device)
            yv = torch.tensor(y_val, dtype=torch.float32).reshape(-1, 1).to(device)
            val_loss = criterion(model(Xv), yv).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= early_stopping:
                logger.info(f" Early stopping на Epoch {epoch+1}")
                break

        if epoch % 5 == 0 or epoch == 0:
            logger.info(f"Epoch {epoch+1}/{epochs}, Val Loss: {val_loss:.6f}")

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    logger.info(f"[OK] MLP успandшно натренований ({len(X_tr)} train withраwithкandв, {epoch+1} епох)")
    return model, best_val_loss


def train_mlp_model(X_train, y_train, task="regression", **kwargs):
    """
    Обгортка for тренування MLP, so that andнтерфейс був як у andнших моwhereлей.
    """
    model, val_loss = train_mlp(X_train, y_train, task=task, **kwargs)
    return model