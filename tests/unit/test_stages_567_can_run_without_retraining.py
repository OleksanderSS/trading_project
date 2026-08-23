"""Verifying twenty minutes of final-stage work cost ten hours of retraining.

Stages 5 to 7 ran for the first time in this project's history on 2026-08-23
and died three minutes in. Finding the NEXT defect would have meant another
`--mode continue`, which always retrained: ten hours to reach twenty minutes of
work. That cost is the reason those stages were the least-tested part of the
system, and it is a defect in the loop rather than in any one line.

`--skip-training` reuses the champions already on disk. They are loaded from
`light_models_results.json` moments earlier anyway -- `_prefer_latest_local_run`
takes the newest run out of it -- so nothing extra is read and nothing is
invented.

The shape returned must match what training produces, or the difference shows
up somewhere in stages 5 to 7 instead of here.
"""

from __future__ import annotations

from src.cli.pipeline_executor import PipelineExecutor


def test_champions_on_disk_are_handed_on_in_the_trainer_s_own_shape():
    champions = {f"AAPL_1d_target_{i}_TREND": {"winner": "catboost"} for i in range(97)}
    out = PipelineExecutor._reuse_trained_champions({"models_metadata": champions})

    assert out["status"] == "light_models_complete", (
        "a different status would make downstream treat this as a failed run"
    )
    assert out["models_metadata"] == champions
    assert "metrics" in out, "run_light_models returns metrics; so must this"
    assert out["reused_from_disk"] is True


def test_an_empty_disk_fails_loudly_rather_than_running_on_nothing():
    """Silently proceeding would send stages 5-7 an empty champion set.

    They would then produce no predictions and report success, which is the
    exact shape of failure this project keeps finding: a stage that returns {}
    after logging, while the orchestrator reads the return value.
    """
    out = PipelineExecutor._reuse_trained_champions({"models_metadata": {}})
    assert out["status"] == "failed"
    assert out["reason"] == "no_champions_on_disk"

    out = PipelineExecutor._reuse_trained_champions({})
    assert out["status"] == "failed"

    out = PipelineExecutor._reuse_trained_champions(None)
    assert out["status"] == "failed"


def test_the_flag_exists_and_defaults_to_training():
    """Skipping training must be asked for; a stale champion set is the risk."""
    from src.cli.argument_parser import create_argument_parser

    parser = create_argument_parser()
    default = parser.parse_args(["--mode", "continue", "--batch-name", "b"])
    assert default.skip_training is False

    asked = parser.parse_args(
        ["--mode", "continue", "--batch-name", "b", "--skip-training"]
    )
    assert asked.skip_training is True


def test_continue_mode_branches_on_the_flag():
    """Wiring, not arithmetic. The helper being correct proves nothing alone."""
    import ast
    import inspect
    import textwrap

    source = textwrap.dedent(
        inspect.getsource(PipelineExecutor.execute_continue_mode)
    )
    tree = ast.parse(source)
    names = {
        ast.unparse(node.func)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
    }
    assert any("_reuse_trained_champions" in name for name in names), (
        "continue mode never calls the reuse path, so the flag does nothing"
    )
    assert any("_run_light_training_for_continue" in name for name in names), (
        "the training path is gone; --skip-training must be a choice, not the "
        "only behaviour"
    )
