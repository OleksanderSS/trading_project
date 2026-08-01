"""A stage that produced nothing must not be reported as a success.

_execute_stage validated the output, and then:

    if validated_output:
        stage_outputs.update(validated_output)
    ...
    return {'status': 'success', 'outputs': validated_output or {}}

so a stage returning {} fell through both branches and still reported success.
The pipeline continued on the PREVIOUS stage's outputs and finished with
"Pipeline execution completed successfully."

That path is reachable and already written into the stages:

- ModelingStage.run returns {} after logging "Enriched data not found.
  Skipping Modeling Stage." -- a run that trained nothing reported success.
- ProcessingStage.run returns {} when raw_data is empty.
- CollectionStage.run aborted with {'raw_data': {}} when a preset yielded no
  tickers. That value nested one level deeper than the success path (which
  returns a flat {data_type: DataFrame} map) AND was truthy, so it passed
  every emptiness check and only surfaced downstream.

Neither 'pipeline_status' nor 'failed_stage' -- the keys _handle_stage_error
writes -- is read anywhere in the codebase, so returning a status was never
going to make a failure visible. Raising is what surfaces.
"""
from __future__ import annotations

import pytest

from src.core.exceptions import DataProcessingError
from src.pipeline.pipeline_orchestrator import PipelineOrchestrator


class _Stage:
    """Stand-in for a pipeline stage: named like the real one, returns what
    the test tells it to."""

    def __init__(self, name, output):
        self.__class__ = type(name, (_Stage,), {})
        self._output = output
        self.ran = False

    async def run(self, **kwargs):
        self.ran = True
        return self._output


def _orchestrator(stages):
    """A PipelineOrchestrator with the collaborators run() touches stubbed.

    Built without __init__ on purpose: the real constructor loads config,
    builds an ErrorHandler, a DataManager and a HealthHub, none of which this
    contract depends on.
    """
    import logging

    orch = object.__new__(PipelineOrchestrator)
    orch.stages = stages
    orch.stages_to_run = None
    orch.logger = logging.getLogger("test_orchestrator")
    orch.error_handler = _NullErrorHandler()
    orch.memory_profiler = _NullProfiler()
    orch.health_hub = None
    return orch


class _NullErrorHandler:
    def handle_error(self, *args, **kwargs):
        return {}


class _NullProfiler:
    def track(self, _label):
        from contextlib import nullcontext
        return nullcontext()


@pytest.fixture(autouse=True)
def _stub_environment(monkeypatch):
    monkeypatch.setattr(PipelineOrchestrator, "_get_memory_usage", lambda self: 0.0)
    monkeypatch.setattr(PipelineOrchestrator, "_log_memory_statistics", lambda self: None)
    monkeypatch.setattr(PipelineOrchestrator, "_log_models_metadata",
                        lambda self, name, outputs: None)
    monkeypatch.setattr(PipelineOrchestrator, "_initialize_stage_outputs",
                        lambda self, ctx: dict(ctx))
    # The schema check is a separate concern; this test is about emptiness.
    monkeypatch.setattr(PipelineOrchestrator, "_validate_stage_output",
                        PipelineOrchestrator._validate_stage_output)


REQUIRED = ["CollectionStage", "ProcessingStage", "FeatureEngineeringStage", "ModelingStage"]
OPTIONAL = ["Stage0Setup", "PredictionStage", "TradingExecutionStage", "EvaluationStage"]


@pytest.mark.parametrize("stage_name", REQUIRED)
@pytest.mark.parametrize("empty", [{}, None], ids=["empty-dict", "none"])
def test_a_required_stage_producing_nothing_is_not_a_success(stage_name, empty):
    orch = _orchestrator([_Stage(stage_name, empty)])

    with pytest.raises(RuntimeError) as raised:
        orch._execute_sync(orch.run())

    assert stage_name in str(raised.value)


@pytest.mark.parametrize("stage_name", OPTIONAL)
def test_an_optional_stage_may_legitimately_produce_nothing(stage_name):
    """Stage0Setup has no data output; an empty prediction set is documented
    as an expected outcome of the champion filter."""
    orch = _orchestrator([_Stage(stage_name, {})])

    assert orch._execute_sync(orch.run()) is not None


def test_the_pipeline_does_not_run_on_after_a_stage_produced_nothing():
    """The concrete regression: modeling skips, and everything after it used
    to proceed as if a model had been trained."""
    modeling = _Stage("ModelingStage", {})
    prediction = _Stage("PredictionStage", {"predictions": ["should never happen"]})
    orch = _orchestrator([modeling, prediction])

    with pytest.raises(RuntimeError):
        orch._execute_sync(orch.run())

    assert modeling.ran
    assert not prediction.ran, "a stage ran on data the previous stage never produced"


def test_output_still_flows_through_when_stages_produce_data():
    # ModelingStage and EvaluationStage carry no pydantic schema; the schema'd
    # stages are exercised by their own validation tests, and a toy payload
    # here would only be testing pydantic.
    orch = _orchestrator([
        _Stage("ModelingStage", {"models_meta": {"AAPL": "model"}}),
        _Stage("EvaluationStage", {"report": {"sharpe": 1.2}}),
    ])

    outputs = orch._execute_sync(orch.run())

    assert outputs["models_meta"] == {"AAPL": "model"}
    assert outputs["report"] == {"sharpe": 1.2}


def test_validation_names_the_stage_and_points_at_its_log():
    orch = _orchestrator([])

    with pytest.raises(DataProcessingError, match="ModelingStage produced no output"):
        orch._validate_stage_output("ModelingStage", {})


def test_collection_abort_matches_the_shape_of_its_success_path():
    """CollectionStage.run returned {'raw_data': {}} when a preset had no
    tickers, while the success path returns a flat {data_type: DataFrame}
    map that the orchestrator assigns straight to stage_outputs['raw_data'].
    The nested value was also truthy, so it slipped past every check."""
    import inspect

    from src.pipeline.stages.collection import orchestrator as collection

    source = inspect.getsource(collection.CollectionStage.run)
    code = "\n".join(
        line for line in source.splitlines() if not line.strip().startswith("#")
    )
    assert "{'raw_data': {}}" not in code
    assert '{"raw_data": {}}' not in code
