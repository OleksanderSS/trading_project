from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SCAFFOLD = str(Path(__file__).parent.parent.parent / "dean_domain_scaffold.py")


def _run(*args: str) -> tuple[int, str]:
    result = subprocess.run(
        [sys.executable, SCAFFOLD, *args],
        capture_output=True,
        text=True,
        timeout=60,
    )
    return result.returncode, result.stdout + result.stderr


def test_list():
    rc, out = _run("list")
    assert rc == 0, out
    assert "semiconductor_ai_infrastructure" in out


def test_list_agents():
    rc, out = _run("list-agents")
    assert rc == 0, out
    assert "38 agents" in out or "Total:" in out


def test_registry_show():
    rc, out = _run("registry", "show", "pipeline_readiness")
    assert rc == 0, out
    assert "PipelineReadinessAgent" in out


def test_profiles_show():
    rc, out = _run("profiles", "show", "semiconductor_ai_infrastructure")
    assert rc == 0, out
    assert "Semiconductors" in out


def test_validate_config():
    rc, out = _run("validate-config")
    assert rc == 0, out
    assert "valid" in out or "issue" in out


def test_health():
    rc, out = _run("health")
    assert rc == 0, out
    assert "duckdb" in out.lower() or "OK" in out


def test_health_json():
    rc, out = _run("health", "--json")
    assert rc == 0, out
    assert '"checks"' in out


def test_stats():
    rc, out = _run("stats")
    assert rc == 0, out


def test_search():
    rc, out = _run("search", "ticker")
    assert rc == 0, out
    assert "ticker" in out


def test_help():
    rc, out = _run("--help")
    assert rc == 0, out
    assert "Usage:" in out


def test_unknown_command():
    rc, out = _run("nonexistent_command")
    assert rc == 1
