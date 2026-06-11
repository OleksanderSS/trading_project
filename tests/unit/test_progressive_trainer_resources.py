from types import SimpleNamespace
from unittest.mock import patch

from src.training.progressive_trainer import ProgressiveTrainer


class DummyLogger:
    def warning(self, *_args, **_kwargs):
        return None

    def debug(self, *_args, **_kwargs):
        return None


class DummyMemory:
    used = 10 * 1024 ** 3
    percent = 50.0


def make_trainer():
    trainer = ProgressiveTrainer.__new__(ProgressiveTrainer)
    trainer.config = SimpleNamespace(max_time_hours=1, max_memory_gb=8.0)
    trainer.state_manager = SimpleNamespace(state=SimpleNamespace(start_time=0.0))
    trainer.logger = DummyLogger()
    return trainer


def test_progressive_trainer_blocks_when_time_budget_exceeded():
    trainer = make_trainer()

    with patch("src.training.progressive_trainer.time.time", return_value=7200.0):
        assert trainer._check_resources() is False


def test_progressive_trainer_blocks_when_memory_budget_exceeded():
    trainer = make_trainer()

    with patch("src.training.progressive_trainer.time.time", return_value=10.0):
        with patch("psutil.virtual_memory", return_value=DummyMemory()):
            with patch("psutil.cpu_percent", return_value=10.0):
                assert trainer._check_resources() is False
