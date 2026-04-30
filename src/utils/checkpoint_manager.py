"""
Управління контрольними точками моделей
"""
import os
import json
from pathlib import Path
from typing import Optional, Dict, Any
from src.config.training_config import CheckpointParams


class CheckpointManager:
    """Управління збереженням та завантаженням контрольних точок"""

    @staticmethod
    def save_checkpoint(params: CheckpointParams) -> None:
        """Зберегти контрольну точку"""
        os.makedirs(os.path.dirname(params.model_path), exist_ok=True)
        
        checkpoint_data = {
            'model_path': params.model_path,
            'scaler_path': params.scaler_path,
            'epoch': params.epoch
        }
        
        if params.optimizer_state:
            checkpoint_data['optimizer_state'] = params.optimizer_state

        checkpoint_file = params.model_path.replace('.pt', '_checkpoint.json')
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)

    @staticmethod
    def load_checkpoint(checkpoint_path: str, model: object, optimizer: object) -> Dict[str, Any]:
        """Завантажити контрольну точку"""
        if not os.path.exists(checkpoint_path):
            return {'epoch': 0, 'best_loss': float('inf')}

        try:
            import torch
            checkpoint = torch.load(checkpoint_path)
            
            if model is not None:
                model.load_state_dict(checkpoint.get('model_state', {}))
            
            if optimizer is not None and 'optimizer_state' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state'])
            
            return {
                'epoch': checkpoint.get('epoch', 0),
                'best_loss': checkpoint.get('best_loss', float('inf'))
            }
        except Exception as e:
            print(f"Помилка при завантаженні контрольної точки: {e}")
            return {'epoch': 0, 'best_loss': float('inf')}

    @staticmethod
    def find_latest_checkpoint(checkpoint_dir: str, ticker: str, target_col: str, 
                              model_type: str) -> Optional[str]:
        """Знайти останню контрольну точку"""
        if not os.path.exists(checkpoint_dir):
            return None

        pattern = f"{ticker}_{target_col}_{model_type}_*.pt"
        checkpoints = list(Path(checkpoint_dir).glob(pattern))
        
        if not checkpoints:
            return None

        return str(max(checkpoints, key=lambda p: p.stat().st_mtime))
