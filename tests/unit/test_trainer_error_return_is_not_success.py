"""A trainer that RETURNS an error was recorded as a success.

Several trainers report failure by returning a dict rather than raising:
TabNet when pytorch_tabnet is missing, the sequence models when there is
not enough history to build even one window. The caller looked at none of
that. It wrote status='success', saved a metrics sidecar, and listed a
model_path for a file that was never written.

Found by counting, on the 2026-08-07 Colab run:

    MODELS           : 4619
    metrics sidecars : 4620
    selected features: 4620

One sidecar with no model. Every other model type came out at exactly 660;
tabnet at 659.

It is not a cosmetic mismatch. colab_results.json is what Stage 5 reads to
build its candidate list, so a success entry for a missing file puts a path
to nothing in front of the prediction stage -- and the metrics under it are
{'error': ...}, which scores -inf, so the failure surfaces (if at all) as a
model that merely never wins.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _controller_class():
    path = Path("scripts/colab/colab_clean_cell.py")
    if not path.exists():
        pytest.skip("colab trainer script not present")
    spec = importlib.util.spec_from_file_location("colab_clean_cell", path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"colab trainer imports unavailable here: {exc}")
    return module.ColabTrainingController


def _run(cls, tmp_path, metrics, write_model):
    """Drive the REAL _train_model, stubbing only its collaborators.

    Built as a fake class at first, which meant re-attaching methods to it
    -- and that quietly turned _context_id from a staticmethod into an
    instance method, so the test failed on its own scaffolding rather than
    on the behaviour. The real class, with per-instance stubs, has no such
    seam.
    """
    controller = object.__new__(cls)
    controller.results = {'ticker_results': {}, 'models_metadata': {}}
    controller.sidecars_written = []

    class _Batch:
        batch_dir = tmp_path

    class _Selector:
        @staticmethod
        def select_features(**kwargs):
            return ['f1', 'f2']

    controller.path_manager = _Batch()
    controller.feature_selector = _Selector()

    controller._get_model_max_features = lambda model_type: 10
    controller._batch_fingerprint = lambda: 'fp'
    controller._load_sidecar = lambda name: None
    controller._save_sidecar = (
        lambda name, m, f: controller.sidecars_written.append(name)
    )

    def _train(ticker, target_col, model_type, x_df, y_ser, selected_features,
               target_type, is_classification, y_scaler, model_path):
        if write_model:
            Path(model_path).write_bytes(b'model')
        return metrics

    controller._train_model_with_features = _train

    controller._train_model(
        'AAPL', '1d', 'target_up_1d', 'tabnet',
        x_df=None, y_ser=None, target_type='classification_binary',
        is_classification=True, y_scaler=None,
    )
    return controller


def _models_recorded(recorder):
    return (
        recorder.results['ticker_results']
        .get('AAPL', {}).get('timeframes', {})
        .get('1d', {}).get('results', {})
        .get('target_up_1d', {}).get('models', {})
    )


def test_an_error_return_is_recorded_as_an_error(tmp_path):
    cls = _controller_class()

    recorder = _run(cls, tmp_path,
                    metrics={'error': 'pytorch_tabnet not installed'},
                    write_model=False)

    recorded = _models_recorded(recorder)
    assert recorded['tabnet']['status'] == 'error'
    assert 'pytorch_tabnet' in recorded['tabnet']['message']


def test_an_error_return_writes_no_sidecar(tmp_path):
    """The sidecar is the artifact that outlived the run and produced the
    off-by-one that exposed this."""
    cls = _controller_class()

    recorder = _run(cls, tmp_path,
                    metrics={'error': 'insufficient history'},
                    write_model=False)

    assert recorder.sidecars_written == []


def test_an_error_return_is_not_offered_to_stage_5(tmp_path):
    cls = _controller_class()

    recorder = _run(cls, tmp_path, metrics={'error': 'boom'}, write_model=False)

    assert recorder.results['models_metadata'] == {}, (
        "a failed model was listed as a Stage 5 candidate"
    )


def test_metrics_without_a_model_file_are_refused(tmp_path):
    """Numbers with no artifact behind them are a path to nothing."""
    cls = _controller_class()

    recorder = _run(cls, tmp_path, metrics={'accuracy': 0.61}, write_model=False)

    recorded = _models_recorded(recorder)
    assert recorded['tabnet']['status'] == 'error'
    assert recorder.sidecars_written == []
    assert recorder.results['models_metadata'] == {}


def test_a_real_success_still_records_normally(tmp_path):
    """The guard must not swallow the working path -- the failure mode of
    the previous fix in this file was exactly that shape."""
    cls = _controller_class()

    recorder = _run(cls, tmp_path, metrics={'accuracy': 0.61}, write_model=True)

    recorded = _models_recorded(recorder)
    assert recorded['tabnet']['status'] == 'success'
    assert recorded['tabnet']['metrics'] == {'accuracy': 0.61}
    assert len(recorder.sidecars_written) == 1
    assert len(recorder.results['models_metadata']) == 1
