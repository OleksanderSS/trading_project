"""
Управління контрольними точками моделей
"""
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.utils.artifact_security import resolve_trusted_artifact_path


@dataclass
class CheckpointParams:
    """Parameters for checkpoint saving"""
    model_path: str
    scaler_path: str
    epoch: int
    optimizer_state: Any = None

_logger = logging.getLogger(__name__)


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
    def load_checkpoint(checkpoint_path: str, model: object, optimizer: object) -> dict[str, Any]:
        """Завантажити контрольну точку"""
        if not os.path.exists(checkpoint_path):
            return {'epoch': 0, 'best_loss': float('inf')}

        try:
            import torch
            trusted_checkpoint_path = resolve_trusted_artifact_path(
                checkpoint_path,
                allowed_suffixes={'.pt', '.pth'},
                must_exist=True,
            )
            # SEC-3: weights_only=True prevents arbitrary code execution from
            # malicious checkpoint files (PyTorch security advisory, v2.0+)
            checkpoint = torch.load(  # audit-ignore: UNSAFE_MODEL_OR_PICKLE_LOAD
                trusted_checkpoint_path, weights_only=True)

            if model is not None:
                model.load_state_dict(checkpoint.get('model_state', {}))  # type: ignore[attr-defined]

            if optimizer is not None and 'optimizer_state' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state'])  # type: ignore[attr-defined]

            return {
                'epoch': checkpoint.get('epoch', 0),
                'best_loss': checkpoint.get('best_loss', float('inf'))
            }
        except Exception as e:
            _logger.error(f"Помилка при завантаженні контрольної точки: {e}")
            return {'epoch': 0, 'best_loss': float('inf')}

    @staticmethod
    def find_latest_checkpoint(checkpoint_dir: str, ticker: str, target_col: str,
                              model_type: str) -> str | None:
        """Знайти останню контрольну точку"""
        if not os.path.exists(checkpoint_dir):
            return None

        pattern = f"{ticker}_{target_col}_{model_type}_*.pt"
        checkpoints = list(Path(checkpoint_dir).glob(pattern))

        if not checkpoints:
            return None

        return str(max(checkpoints, key=lambda p: p.stat().st_mtime))
