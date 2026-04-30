"""Checkpoint management for model training"""

from pathlib import Path

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False


def save_checkpoint(params):
    """Зберегти checkpoint для відновлення тренування"""
    if not TORCH_AVAILABLE:
        print("   ⚠️ torch не доступний, checkpoint не збережено")
        return None
        
    checkpoint_path = Path(params.checkpoint_dir) / \
        f"checkpoint_{params.ticker}_{params.target_col}_{params.m_type}_ep{params.epoch}.pt"
    torch.save({
        'model_state_dict': params.model.state_dict(),
        'optimizer_state_dict': params.optimizer.state_dict(),
        'epoch': params.epoch,
        'loss': params.loss
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
    pattern = f"checkpoint_{ticker}_{target_col}_{m_type}_ep*.pt"
    checkpoints = list(checkpoint_dir.glob(pattern))
    if checkpoints:
        # Сортуємо за номером епохи (спадаючо)
        checkpoints.sort(key=lambda x: int(
            x.stem.split('_ep')[-1]), reverse=True)
        return checkpoints[0]
    return None
