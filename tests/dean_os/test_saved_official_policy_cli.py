import pytest
import sys
from run_agent_saved_official_policy_evidence import main

def test_parser_help_smoke(capsys, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_agent_saved_official_policy_evidence.py", "--help"])
    with pytest.raises(SystemExit) as e:
        main()
    assert e.value.code == 0
    captured = capsys.readouterr()
    assert "Bind one official policy source to independent news corroboration." in captured.out

def test_cli_arguments_passing(monkeypatch):
    class MockProducer:
        def __init__(self, output_dir):
            self.output_dir = output_dir

        def build(self, snapshot_artifact_path, corroborating_news_artifact_path, as_of, registry_path, save):
            assert snapshot_artifact_path == "snap.json"
            assert corroborating_news_artifact_path == "news.json"
            assert as_of == "2026-07-01T00:00:00+00:00"
            assert registry_path == "custom_registry.yaml"
            assert save is True
            return {"run_id": "test_run_123"}

    monkeypatch.setattr(
        "run_agent_saved_official_policy_evidence.SavedOfficialPolicyEvidenceProducer",
        MockProducer
    )

    test_args = [
        "run_agent_saved_official_policy_evidence.py",
        "snap.json",
        "news.json",
        "--as-of", "2026-07-01T00:00:00+00:00",
        "--registry-path", "custom_registry.yaml",
        "--output-dir", "test_output_dir"
    ]
    monkeypatch.setattr(sys, "argv", test_args)
    main()

def test_cli_no_save_flag(monkeypatch, capsys):
    class MockProducer:
        def __init__(self, output_dir):
            pass

        def build(self, snapshot_artifact_path, corroborating_news_artifact_path, as_of, registry_path, save):
            assert save is False
            return {}

    monkeypatch.setattr(
        "run_agent_saved_official_policy_evidence.SavedOfficialPolicyEvidenceProducer",
        MockProducer
    )

    test_args = [
        "run_agent_saved_official_policy_evidence.py",
        "snap.json",
        "news.json",
        "--as-of", "2026-07-01T00:00:00+00:00",
        "--no-save"
    ]
    monkeypatch.setattr(sys, "argv", test_args)
    main()
    captured = capsys.readouterr()
    assert "Successfully verified official policy (no-save)." in captured.out
