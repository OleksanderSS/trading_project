from types import SimpleNamespace

import pytest

from src.cli.argument_validator import ArgumentValidator


class DummyConfigManager:
    def get_config(self, key):
        return {}


def test_argument_validator_rejects_continue_without_batch_name():
    args = SimpleNamespace(
        mode='continue',
        batch_name=None,
        test_ticker=None,
        test_target=None,
        test_model=None,
        max_iterations=None,
        stages=None
    )
    config_manager = DummyConfigManager()

    with pytest.raises(ValueError, match='Argument validation failed'):
        ArgumentValidator.validate_arguments(args, config_manager)
