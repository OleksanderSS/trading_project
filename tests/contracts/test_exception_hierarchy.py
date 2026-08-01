"""One name, one exception class.

Two modules each defined a PipelineError and a ConfigurationError, and the
pairs were unrelated:

    src/core/exceptions.py            PipelineError(Exception)
                                      ConfigurationError(PipelineError)
    src/core/error_handling/…         TradingSystemError(Exception)
                                      PipelineError(TradingSystemError)
                                      ConfigurationError(TradingSystemError)

`except ConfigurationError` therefore caught only whichever class the
catching module imported, and a raise of the other passed straight through
the handler. Verified as latent rather than live at the time: the one raise
and the one except sit in the same file behind the same import.

The hierarchy is now defined once, in src/core/exceptions.py, and
error_handler re-exports it so existing imports keep working.
"""
from __future__ import annotations

import pytest

from src.core import exceptions as canonical
from src.core.error_handling import error_handler

SHARED_NAMES = [
    "TradingSystemError",
    "PipelineError",
    "ConfigurationError",
    "StageError",
    "StageExecutionError",
    "ModelLoadingError",
]


@pytest.mark.parametrize("name", SHARED_NAMES)
def test_both_modules_expose_the_same_class_object(name):
    assert getattr(error_handler, name) is getattr(canonical, name), (
        f"{name} is defined twice; a handler catches only the one it imported"
    )


@pytest.mark.parametrize("name", [
    "PipelineError",
    "DataLoadError",
    "DataProcessingError",
    "ModelTrainingError",
    "FeatureSelectionError",
    "ConfigurationError",
    "StageError",
    "StageExecutionError",
    "ModelLoadingError",
])
def test_every_domain_error_descends_from_the_single_root(name):
    assert issubclass(getattr(canonical, name), canonical.TradingSystemError)


def test_catching_the_base_catches_the_specific_ones():
    """What the split hierarchy made impossible."""
    for error in (
        canonical.DataProcessingError("x"),
        canonical.DataLoadError("x"),
        canonical.ConfigurationError("x"),
        canonical.ModelLoadingError("x"),
    ):
        try:
            raise error
        except canonical.PipelineError:
            pass
        else:  # pragma: no cover
            pytest.fail(f"{type(error).__name__} escaped `except PipelineError`")


def test_a_configuration_error_raised_anywhere_is_caught_anywhere():
    """The exact cross-module case that used to fall through."""
    from src.core.error_handling.error_handler import (
        ConfigurationError as FromHandler,
    )
    from src.core.exceptions import ConfigurationError as FromCanonical

    with pytest.raises(FromCanonical):
        raise FromHandler("raised via one import, caught via the other")


def test_the_hierarchy_lives_in_one_module():
    """error_handler must import these, not define them again."""
    import inspect

    source = inspect.getsource(error_handler)
    code = "\n".join(
        line for line in source.splitlines() if not line.strip().startswith("#")
    )

    for name in SHARED_NAMES:
        assert f"class {name}(" not in code, (
            f"{name} is defined in error_handler again"
        )
