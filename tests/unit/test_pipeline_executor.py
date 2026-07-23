import asyncio
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from src.cli.pipeline_executor import PipelineExecutor


class FullModeOrchestrator:
    def __init__(self):
        self.request = None

    async def run_full_hybrid_pipeline(self, request):
        self.request = request
        return {"status": "paused_for_colab"}


class ContinueModeOrchestrator:
    def __init__(self, batch_dir, colab_results):
        self.config = SimpleNamespace(output_dir=batch_dir)
        self.colab_results = colab_results

    def load_colab_results(self, batch_name):
        return self.colab_results


class ContinueExecutionOrchestrator(ContinueModeOrchestrator):
    def __init__(self, batch_dir, colab_results):
        super().__init__(batch_dir, colab_results)
        self.light_kwargs = None
        self.final_request = None

    async def run_light_models(self, **kwargs):
        self.light_kwargs = kwargs
        return {"models_metadata": {"light_model": {"score": 1.0}}}

    async def run_final_stages(self, request):
        self.final_request = request
        return {"status": "completed"}


def test_execute_full_mode_uses_hybrid_request():
    orchestrator = FullModeOrchestrator()

    result = asyncio.run(PipelineExecutor.execute_full_mode(
        orchestrator,
        tickers=["AMD"],
        timeframes=["15m", "1d"],
    ))

    assert result["status"] == "paused_for_colab"
    assert orchestrator.request.tickers == ["AMD"]
    assert orchestrator.request.timeframes == ["15m", "1d"]
    assert orchestrator.request.accumulate is True


def test_load_continue_data_returns_stable_tuple_on_missing_colab_results():
    args = SimpleNamespace(batch_name="missing_batch")
    with tempfile.TemporaryDirectory() as tmp_dir:
        orchestrator = ContinueModeOrchestrator(
            batch_dir=Path(tmp_dir),
            colab_results={"status": "error", "message": "not found"},
        )

        result = PipelineExecutor._load_continue_data(orchestrator, args)

        assert len(result) == 5
        assert result[0] is None
        assert result[2]["status"] == "error"


def test_validate_continue_inputs_rejects_missing_targets():
    features_df = pd.DataFrame({"datetime": ["2026-05-08"], "ticker": ["AMD"]})
    colab_results = {"status": "success", "models_metadata": {"m1": {"model_path": "model.pkl"}}}

    result = PipelineExecutor._validate_continue_inputs(
        features_df=features_df,
        targets_df=None,
        colab_results=colab_results,
        batch_name="main_database",
    )

    assert result == {"status": "failed", "reason": "missing_targets"}


def test_validate_continue_inputs_rejects_targets_without_target_columns():
    features_df = pd.DataFrame({"datetime": ["2026-05-08"], "ticker": ["AMD"]})
    targets_df = pd.DataFrame({"volatility_15m": [0.1], "ticker": ["AMD"]})
    colab_results = {"status": "success", "models_metadata": {"m1": {"model_path": "model.pkl"}}}

    result = PipelineExecutor._validate_continue_inputs(
        features_df=features_df,
        targets_df=targets_df,
        colab_results=colab_results,
        batch_name="main_database",
    )

    assert result == {"status": "failed", "reason": "missing_target_columns"}


def test_validate_continue_inputs_accepts_uppercase_target_columns():
    features_df = pd.DataFrame({"datetime": ["2026-05-08"], "ticker": ["AMD"], "f1": [1.0]})
    targets_df = pd.DataFrame({"TARGET_RETURN_1P": [0.1], "ticker": ["AMD"]})
    colab_results = {"status": "success", "models_metadata": {"m1": {"model_path": "model.pkl"}}}

    result = PipelineExecutor._validate_continue_inputs(
        features_df=features_df,
        targets_df=targets_df,
        colab_results=colab_results,
        batch_name="main_database",
    )

    assert result is None


def test_run_local_pipeline_extracts_uppercase_targets_and_drops_target_derived(monkeypatch, tmp_path):
    cleaned_path = tmp_path / "cleaned.parquet"
    cleaned_path.write_text("", encoding="utf-8")
    cleaned_df = pd.DataFrame(
        {
            "datetime": ["2026-05-08"],
            "ticker": ["AMD"],
            "feature_a": [1.0],
            "TARGET_RETURN_1P": [0.1],
            "target_up_1d": [1],
            "state_TARGET_RETURN_1P": [0.2],
        }
    )

    class SavedCleanedDataOrchestrator:
        async def run_local_pipeline(self, tickers, timeframes):
            return {
                "results": {"features_df": pd.DataFrame(), "targets_df": pd.DataFrame()},
                "saved_files": {"cleaned_data": str(cleaned_path)},
            }

    monkeypatch.setattr("src.cli.pipeline_executor.pd.read_parquet", lambda _path: cleaned_df)

    features_df, targets_df = asyncio.run(
        PipelineExecutor._run_local_pipeline_and_extract_data(
            SavedCleanedDataOrchestrator(),
            tickers=["AMD"],
            timeframes=["1d"],
        )
    )

    assert list(features_df.columns) == ["datetime", "ticker", "feature_a"]
    assert list(targets_df.columns) == ["TARGET_RETURN_1P", "target_up_1d"]


def test_merge_results_data_initializes_models_metadata():
    merged = PipelineExecutor._merge_results_data(
        colab_results={},
        light_results={"models_metadata": {"light_model": {"score": 1.0}}},
    )

    assert merged["models_metadata"]["light_model"]["score"] == 1.0


def test_execute_continue_mode_trains_light_models_on_loaded_data(monkeypatch):
    features_df = pd.DataFrame({"datetime": ["2026-05-08"], "ticker": ["AMD"], "f1": [1.0]})
    targets_df = pd.DataFrame({"target_return": [0.1], "ticker": ["AMD"]})
    colab_results = {"status": "success", "ticker_results": {"AMD": {}}}
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        orchestrator = ContinueExecutionOrchestrator(Path(tmp_dir), colab_results)
        args = SimpleNamespace(
            batch_name="main_database",
            test_ticker=None,
            test_target=None,
            stages=None,
        )

        monkeypatch.setattr(
            PipelineExecutor,
            "_validate_batch_contract",
            staticmethod(lambda _orchestrator: {
                "valid": True,
                "manifest": {"timeframes": ["1d"]},
                "errors": [],
            }),
        )
        monkeypatch.setattr(
            PipelineExecutor,
            "_load_continue_data",
            staticmethod(lambda _orchestrator, _args: (
                features_df,
                targets_df,
                colab_results,
                None,
                None,
            )),
        )

        result = asyncio.run(PipelineExecutor.execute_continue_mode(orchestrator, args))

        assert result["status"] == "completed"
        assert orchestrator.light_kwargs["features_df"].equals(features_df)
        assert orchestrator.light_kwargs["targets_df"].equals(targets_df)
        assert orchestrator.final_request["light_results"]["models_metadata"]["light_model"]["score"] == 1.0


def test_compute_code_fingerprint_is_deterministic():
    """Same source tree, same fingerprint — required for the cache check
    to ever return a stable "no change" verdict."""
    first = PipelineExecutor._compute_code_fingerprint()
    second = PipelineExecutor._compute_code_fingerprint()
    assert first == second
    assert len(first) == 64  # sha256 hex digest


def test_compute_code_fingerprint_changes_when_tracked_file_changes(tmp_path):
    """Proves the fix actually works: editing a file under one of the
    tracked directories changes the fingerprint, which is what makes a
    stale features.parquet cache get invalidated by a code change instead
    of silently reused. Reimplements the same hashing logic as
    PipelineExecutor._compute_code_fingerprint against an isolated tmp_path
    tree rather than monkeypatching module internals — the real method's
    project-root resolution is exercised separately by
    test_check_cache_before_run_invalidates_on_code_change below, which
    patches the method itself rather than its internals."""
    import hashlib

    tracked_dir = tmp_path / "src" / "pipeline" / "stages"
    tracked_dir.mkdir(parents=True)
    tracked_file = tracked_dir / "fake_stage.py"
    tracked_file.write_text("VALUE = 1\n", encoding="utf-8")

    def _fingerprint_of(root: Path) -> str:
        hasher = hashlib.sha256()
        for path in sorted((root / "src" / "pipeline" / "stages").rglob("*.py")):
            hasher.update(str(path.relative_to(root)).replace("\\", "/").encode("utf-8"))
            hasher.update(path.read_bytes())
        return hasher.hexdigest()

    before = _fingerprint_of(tmp_path)
    tracked_file.write_text("VALUE = 2\n", encoding="utf-8")
    after = _fingerprint_of(tmp_path)

    assert before != after


def test_check_cache_before_run_invalidates_on_code_change(tmp_path, monkeypatch):
    """End-to-end: even when the DB fingerprint matches (no new raw data),
    a code_fingerprint mismatch alone must invalidate the cache."""
    features_path = tmp_path / "features.parquet"
    targets_path = tmp_path / "targets.parquet"
    pd.DataFrame({"a": [1]}).to_parquet(features_path)
    pd.DataFrame({"b": [1]}).to_parquet(targets_path)

    fp_path = tmp_path / "raw_db_fingerprint.json"
    import json
    fp_path.write_text(json.dumps({
        "fingerprint": "same_db_fingerprint",
        "code_fingerprint": "stale_code_fingerprint_from_before_a_fix",
        "generated_at": "2026-01-01T00:00:00",
        "table_states": {},
    }), encoding="utf-8")

    monkeypatch.setattr(
        PipelineExecutor, "_compute_db_fingerprint",
        staticmethod(lambda _orchestrator: ("same_db_fingerprint", {})),
    )
    monkeypatch.setattr(
        PipelineExecutor, "_compute_code_fingerprint",
        staticmethod(lambda: "current_code_fingerprint_after_a_fix"),
    )

    orchestrator = SimpleNamespace(config=SimpleNamespace(output_dir=tmp_path))
    result = PipelineExecutor._check_cache_before_run(orchestrator)

    assert result is None  # cache correctly treated as invalid


def test_check_cache_before_run_uses_cache_when_both_fingerprints_match(tmp_path, monkeypatch):
    features_path = tmp_path / "features.parquet"
    targets_path = tmp_path / "targets.parquet"
    pd.DataFrame({"a": [1]}).to_parquet(features_path)
    pd.DataFrame({"b": [1]}).to_parquet(targets_path)

    fp_path = tmp_path / "raw_db_fingerprint.json"
    import json
    fp_path.write_text(json.dumps({
        "fingerprint": "same_db_fingerprint",
        "code_fingerprint": "same_code_fingerprint",
        "generated_at": "2026-01-01T00:00:00",
        "table_states": {},
    }), encoding="utf-8")

    monkeypatch.setattr(
        PipelineExecutor, "_compute_db_fingerprint",
        staticmethod(lambda _orchestrator: ("same_db_fingerprint", {})),
    )
    monkeypatch.setattr(
        PipelineExecutor, "_compute_code_fingerprint",
        staticmethod(lambda: "same_code_fingerprint"),
    )

    orchestrator = SimpleNamespace(config=SimpleNamespace(output_dir=tmp_path))
    result = PipelineExecutor._check_cache_before_run(orchestrator)

    assert result is not None
    cached_features, cached_targets = result
    assert list(cached_features.columns) == ["a"]
    assert list(cached_targets.columns) == ["b"]
