import os
import subprocess
import sys
import tempfile
from unittest.mock import patch

from run_agent_strategy_replay_candidate_assessment import main as main_assessment
from run_agent_strategy_maturity_daily_reconciliation import main as main_reconciliation


def test_replay_assessment_cli_imports():
    result = subprocess.run([sys.executable, "run_agent_strategy_replay_candidate_assessment.py", "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "Evaluate one real reviewed hypothesis" in result.stdout

def test_daily_reconciliation_cli_imports():
    result = subprocess.run([sys.executable, "run_agent_strategy_maturity_daily_reconciliation.py", "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "Reconcile a candidate playbook" in result.stdout

def test_replay_assessment_cli_missing_args():
    result = subprocess.run([sys.executable, "run_agent_strategy_replay_candidate_assessment.py"], capture_output=True, text=True)
    assert result.returncode != 0
    assert "the following arguments are required: --review-gate" in result.stderr

def test_daily_reconciliation_cli_missing_args():
    result = subprocess.run([sys.executable, "run_agent_strategy_maturity_daily_reconciliation.py"], capture_output=True, text=True)
    assert result.returncode != 0
    assert "the following arguments are required: --assessment" in result.stderr

def test_replay_assessment_cli_forbidden_args():
    result = subprocess.run([sys.executable, "run_agent_strategy_replay_candidate_assessment.py", "--review-gate", "gate.json", "--apply-ledger"], capture_output=True, text=True)
    assert result.returncode != 0
    assert "unrecognized arguments: --apply-ledger" in result.stderr
    
    result = subprocess.run([sys.executable, "run_agent_strategy_replay_candidate_assessment.py", "--review-gate", "gate.json", "--apply-journal"], capture_output=True, text=True)
    assert result.returncode != 0
    assert "unrecognized arguments: --apply-journal" in result.stderr

def test_daily_reconciliation_cli_forbidden_args():
    result = subprocess.run([sys.executable, "run_agent_strategy_maturity_daily_reconciliation.py", "--assessment", "assessment.json", "--apply-journal"], capture_output=True, text=True)
    assert result.returncode != 0
    assert "unrecognized arguments: --apply-journal" in result.stderr

def test_unknown_args_rejected():
    result = subprocess.run([sys.executable, "run_agent_strategy_replay_candidate_assessment.py", "--review-gate", "gate.json", "--unknown-flag"], capture_output=True, text=True)
    assert result.returncode != 0
    assert "unrecognized arguments: --unknown-flag" in result.stderr
    
    result = subprocess.run([sys.executable, "run_agent_strategy_maturity_daily_reconciliation.py", "--assessment", "assessment.json", "--unknown-flag"], capture_output=True, text=True)
    assert result.returncode != 0
    assert "unrecognized arguments: --unknown-flag" in result.stderr

@patch("run_agent_strategy_replay_candidate_assessment.StrategyReplayCandidateAssessment.build")
def test_replay_assessment_cli_execution(mock_build):
    mock_build.return_value = {"status": "ok"}
    with patch.object(sys, "argv", ["run_agent", "--review-gate", "gate.json", "--hypothesis-id", "hyp123", "--no-save"]):
        main_assessment()
    mock_build.assert_called_once_with(
        review_gate_path="gate.json",
        hypothesis_id="hyp123",
        apply_ledger=False,
        apply_journal=False,
        save=False
    )

@patch("run_agent_strategy_maturity_daily_reconciliation.StrategyMaturityDailyReconciler.build")
def test_daily_reconciliation_cli_execution(mock_build):
    mock_build.return_value = {"status": "ok"}
    with patch.object(sys, "argv", ["run_agent", "--assessment", "assessment.json", "--no-save"]):
        main_reconciliation()
    mock_build.assert_called_once_with(
        candidate_assessment_path="assessment.json",
        risk_snapshot_path=None,
        apply_journal=False,
        save=False
    )

@patch("run_agent_strategy_replay_candidate_assessment.StrategyReplayCandidateAssessment.build")
def test_replay_assessment_cli_no_side_effects(mock_build):
    mock_build.return_value = {"status": "ok"}
    with tempfile.TemporaryDirectory() as tempdir:
        gate_file = os.path.join(tempdir, "gate.json")
        with open(gate_file, "w") as f:
            f.write("{}")
        with patch.object(sys, "argv", ["run_agent", "--review-gate", gate_file, "--no-save"]):
            original_cwd = os.getcwd()
            try:
                os.chdir(tempdir)
                main_assessment()
                assert not os.path.exists("data")
                assert not os.path.exists("reports")
                assert not os.path.exists("ledger")
                assert not os.path.exists("journal")
                assert not os.path.exists("data/dean_os")
                assert not os.path.exists("reports/dean_os")
            finally:
                os.chdir(original_cwd)

@patch("run_agent_strategy_maturity_daily_reconciliation.StrategyMaturityDailyReconciler.build")
def test_daily_reconciliation_cli_no_side_effects(mock_build):
    mock_build.return_value = {"status": "ok"}
    with tempfile.TemporaryDirectory() as tempdir:
        assessment_file = os.path.join(tempdir, "assessment.json")
        with open(assessment_file, "w") as f:
            f.write("{}")
        with patch.object(sys, "argv", ["run_agent", "--assessment", assessment_file, "--no-save"]):
            original_cwd = os.getcwd()
            try:
                os.chdir(tempdir)
                main_reconciliation()
                assert not os.path.exists("data")
                assert not os.path.exists("reports")
                assert not os.path.exists("ledger")
                assert not os.path.exists("journal")
                assert not os.path.exists("data/dean_os")
                assert not os.path.exists("reports/dean_os")
            finally:
                os.chdir(original_cwd)
