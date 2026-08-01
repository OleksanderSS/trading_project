"""Scalers must not be fitted on data the model has not been split from yet.

Stage 2 fits normalisation scalers like this:

    src/pipeline/stages/processing/orchestrator.py
        fit_scalers=run_mode != 'predict'
    src/pipeline/stages/processing/data_handler.py
        self.normalization_manager.fit_scalers(combined, features_to_normalize)

`combined` is the whole processed dataset. The train/test split happens two
stages later, in Stage 4's purged/embargo CV. So whenever the normalisation
list is non-empty, every feature in it is scaled using a mean and a standard
deviation computed over the TEST period as well -- textbook look-ahead. It
does not announce itself: the model simply scores better than it should, and
only live trading reveals the difference.

This is dormant right now, and only by accident: `processing.normalization`
resolves to {} and `features` to [], so fit_scalers runs over an empty list
and scales nothing. Verified, not assumed.

This test is a tripwire, not a fix. Filling that list is a reasonable thing
for someone to want to do, and the moment they do, the leak opens with no
error and no log. Whoever turns it on has to come here and decide where the
fit belongs -- inside the CV fold, not in Stage 2.

Note on the other leakage machinery, since it is easy to assume it covers
this: TemporalLeakageGuard matches 0 of the 713 real feature names (its
patterns target a naming convention this project does not use), and
FeatureLeakageGuard -- which is real and does work -- is wired only into the
Colab/hybrid path (colab_manager.py), not into the main Stage 3. The load-
bearing protection today is the purged/embargo split in Stage 4.
"""
from __future__ import annotations

import pytest

from src.config.unified_config_manager import get_current_config


@pytest.fixture(scope="module")
def normalization_config():
    processing = get_current_config().get_config("processing") or {}
    return processing.get("normalization", {}) or {}


def test_normalisation_is_either_off_or_deliberately_reviewed(normalization_config):
    features = normalization_config.get("features", []) or []

    assert not features, (
        "processing.normalization.features is no longer empty, which means "
        "Stage 2 now fits scalers on the FULL dataset before Stage 4 splits "
        "it -- the test period's mean and standard deviation leak into "
        "training. Move the fit inside the CV fold (fit on train, transform "
        "on validation) before enabling this, then update this test to "
        "assert the new arrangement. Features found: "
        f"{[f.get('name', f) if isinstance(f, dict) else f for f in features]}"
    )


def test_the_fit_still_happens_where_this_test_thinks_it_does():
    """A tripwire is worthless if the code moves out from under it."""
    import inspect

    from src.pipeline.stages.processing import data_handler, orchestrator

    assert "fit_scalers=run_mode != 'predict'" in inspect.getsource(orchestrator), (
        "Stage 2 no longer fits scalers this way; re-read the docstring above "
        "and re-derive whether the leak this guards against still exists."
    )
    assert "fit_scalers(" in inspect.getsource(data_handler)


def test_prediction_runs_never_refit(monkeypatch):
    """The one part of the current arrangement that is unambiguously right:
    a predict run must reuse the scalers from training, not recompute them."""
    import inspect

    from src.pipeline.stages.processing import orchestrator

    source = inspect.getsource(orchestrator)
    assert "run_mode != 'predict'" in source
