from __future__ import annotations

from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).parent.parent.parent / "dean_os" / "config"
PROFILE_DIR = Path(__file__).parent.parent.parent / "config" / "domain_profiles"


def test_agent_registry_parses():
    path = CONFIG_DIR / "agent_registry.yaml"
    assert path.exists(), f"Missing: {path}"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert "agents" in data
    agents = data["agents"]
    assert len(agents) >= 37, f"Expected >= 37 agents, got {len(agents)}"


def test_all_agents_have_required_fields():
    path = CONFIG_DIR / "agent_registry.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    required = {"class_path", "branch", "veto_level", "enabled", "error_behavior", "timeout_seconds"}
    for name, cfg in data["agents"].items():
        missing = required - set(cfg.keys())
        assert not missing, f"{name}: missing {missing}"


def test_all_class_paths_resolve():
    import importlib
    path = CONFIG_DIR / "agent_registry.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    for name, cfg in data["agents"].items():
        cp = cfg.get("class_path", "")
        assert ":" in cp, f"{name}: invalid class_path '{cp}'"
        module, cls = cp.rsplit(":", 1)
        try:
            mod = importlib.import_module(module)
            assert hasattr(mod, cls), f"{name}: class '{cls}' not found in {module}"
        except ImportError as e:
            raise AssertionError(f"{name}: cannot import {module} — {e}") from e


def test_no_duplicate_agent_names():
    path = CONFIG_DIR / "agent_registry.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    names = list(data["agents"].keys())
    assert len(names) == len(set(names)), "Duplicate agent names found"


def test_all_registered_branches_are_valid():
    path = CONFIG_DIR / "agent_registry.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    valid = {"pipeline", "analytical"}
    for name, cfg in data["agents"].items():
        assert cfg.get("branch") in valid, f"{name}: invalid branch '{cfg.get('branch')}'"


def test_all_veto_levels_are_valid():
    path = CONFIG_DIR / "agent_registry.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    valid = {"none", "soft", "hard"}
    for name, cfg in data["agents"].items():
        assert cfg.get("veto_level") in valid, f"{name}: invalid veto_level '{cfg.get('veto_level')}'"


def test_domain_profiles_parse():
    count = 0
    for f in sorted(PROFILE_DIR.glob("*.yaml")):
        data = yaml.safe_load(f.read_text(encoding="utf-8"))
        assert isinstance(data, dict), f"{f.name}: not a mapping"
        assert "domain_id" in data, f"{f.name}: missing domain_id"
        count += 1
    assert count >= 7, f"Expected >= 7 domain profiles, got {count}"


def test_domain_profiles_have_required_fields():
    required = {"domain_id", "display_name", "description", "required_evidence_types"}
    for f in sorted(PROFILE_DIR.glob("*.yaml")):
        data = yaml.safe_load(f.read_text(encoding="utf-8"))
        missing = required - set(data.keys())
        assert not missing, f"{f.name}: missing {missing}"


def test_other_config_yamls_parse():
    for f in sorted(CONFIG_DIR.glob("*.yaml")):
        if f.name == "agent_registry.yaml":
            continue
        data = yaml.safe_load(f.read_text(encoding="utf-8"))
        assert data is not None, f"{f.name}: empty or invalid YAML"
