"""Utility functions for model training and data processing"""

import hashlib
from pathlib import Path
from functools import wraps
import time

# Try to import torch
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False


def get_optimal_batch_size(memory_percent, base_batch_size=32):
    """
    Розрахувати оптимальний розмір батча на основі доступної пам'яті

    Логіка:
    - Якщо пам'ять < 50%: використовуємо base_batch_size
    - Якщо пам'ять 50-75%: зменшуємо до base_batch_size // 2
    - Якщо пам'ять 75-90%: зменшуємо до base_batch_size // 4
    - Якщо пам'ять > 90%: зменшуємо до base_batch_size // 8 (мінімум 2)
    """
    if memory_percent < 50:
        return base_batch_size
    elif memory_percent < 75:
        return max(base_batch_size // 2, 8)
    elif memory_percent < 90:
        return max(base_batch_size // 4, 4)
    else:
        return max(base_batch_size // 8, 2)


def save_checkpoint(checkpoint_dir, ticker, target_col, m_type, model, optimizer, epoch, loss):
    """Зберегти checkpoint для відновлення тренування"""
    if not TORCH_AVAILABLE:
        print("   ⚠️ torch не доступний, checkpoint не збережено")
        return None
        
    checkpoint_path = Path(checkpoint_dir) / \
        f"checkpoint_{ticker}_{target_col}_{m_type}_ep{epoch}.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss
    }, checkpoint_path)
    print(f"   ✅ Checkpoint saved: {checkpoint_path.name}")
    return checkpoint_path


def load_checkpoint(checkpoint_path, model, optimizer):
    """Завантажити checkpoint для відновлення тренування"""
    if not TORCH_AVAILABLE:
        print("   ⚠️ torch не доступний, checkpoint не завантажено")
        return 0, float('inf')
        
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint.get('loss', float('inf'))
    print(f"   ✅ Checkpoint loaded from epoch {epoch} (loss: {loss:.6f})")
    return epoch, loss


def find_latest_checkpoint(checkpoint_dir, ticker, target_col, m_type):
    """Знайти найновіший checkpoint для моделі"""
    checkpoint_dir = Path(checkpoint_dir)
    pattern = f"checkpoint_{ticker}_{target_col}_{m_type}_ep*.pt"
    checkpoints = list(checkpoint_dir.glob(pattern))
    if checkpoints:
        # Сортуємо за номером епохи (спадаючо)
        checkpoints.sort(key=lambda x: int(
            x.stem.split('_ep')[-1]), reverse=True)
        return checkpoints[0]
    return None


def retry_on_timeout(max_retries=3, wait_seconds=5):
    """Декоратор для повтору при timeout"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except (TimeoutError, ConnectionError, RuntimeError) as e:
                    if attempt < max_retries - 1:
                        print(
                            f"⚠️ Attempt {attempt + 1} failed: "
                            f"{str(e)[:100]}")
                        print(f"   Retrying in {wait_seconds} seconds...")
                        time.sleep(wait_seconds)
                    else:
                        print(f"❌ Failed after {max_retries} attempts")
                        raise
        return wrapper
    return decorator


def compute_data_signature(df_feat, df_targ):
    """Обчислити сигнатуру даних для кешування"""
    import pandas as pd
    feat_info = (
        f"{df_feat.shape}_"
        f"{pd.util.hash_pandas_object(df_feat.tail(100)).sum()}"
    )
    targ_info = (
        f"{df_targ.shape}_"
        f"{pd.util.hash_pandas_object(df_targ.tail(100)).sum()}"
    )
    combined = f"{feat_info}_{targ_info}"
    return hashlib.sha256(combined.encode()).hexdigest()


def compute_metrics(y_true, y_pred):
    """Розрахувати метрики якості моделі"""
    import numpy as np
    from sklearn.metrics import (
        mean_absolute_error,
        mean_squared_error,
        r2_score,
    )

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mask = y_true != 0
    if mask.sum() > 0:
        mape = np.mean(
            np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])
        ) * 100
    else:
        mape = 0.0
    return {
        'mae': float(mae),
        'rmse': float(rmse),
        'r2': float(r2),
        'mape': float(mape)
    }
