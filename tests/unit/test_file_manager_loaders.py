"""The YAML and JSON loaders must fail the same way, and never fail silently.

FileManager is reached by all eight pipeline stages, and UnifiedConfigManager
loads every config file through load_yaml, so its failure modes propagate
everywhere.

Two defects were found:

- malformed YAML escaped as a raw ParserError while malformed JSON came back
  as the intended RuntimeError. yaml.YAMLError inherits from Exception, not
  from ValueError, so the project's usual
  (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) tuple
  missed it -- two loaders in one class disagreeing on the same failure.
- a YAML file that parsed to a list or a scalar returned None in silence,
  which a caller cannot tell apart from "file not found". A config file
  present but shaped wrong was skipped without explanation.
"""
from __future__ import annotations

import json
import logging

import pytest

from src.core.file_management.file_manager import FileManager


@pytest.fixture()
def manager(tmp_path, monkeypatch):
    """FileManager rooted at tmp_path.

    Its path guard refuses anything outside the base directory -- that guard
    works, and is why these fixtures live under the manager's own root.
    """
    return FileManager(base_dir=tmp_path)


def _write(tmp_path, name, content):
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return path


def test_malformed_yaml_raises_the_same_way_as_malformed_json(manager, tmp_path):
    bad_yaml = _write(tmp_path, "bad.yaml", "key: [unclosed\n  other: :\n")
    bad_json = _write(tmp_path, "bad.json", "{not json")

    with pytest.raises(RuntimeError, match="Failed to load YAML"):
        manager.load_yaml(bad_yaml)
    with pytest.raises(RuntimeError, match="Failed to load JSON"):
        manager.load_json(bad_json)


@pytest.mark.parametrize("content,shape", [
    ("- a\n- b\n", "list"),
    ("just a string\n", "str"),
    ("42\n", "int"),
])
def test_non_mapping_yaml_is_reported_not_silently_dropped(
    manager, tmp_path, caplog, content, shape
):
    path = _write(tmp_path, "shape.yaml", content)

    with caplog.at_level(logging.WARNING):
        assert manager.load_yaml(path) is None

    assert any("not a mapping" in r.getMessage() for r in caplog.records), (
        f"a top-level {shape} was dropped without explanation"
    )


def test_non_object_json_is_reported_not_silently_dropped(manager, tmp_path, caplog):
    path = _write(tmp_path, "arr.json", json.dumps([1, 2, 3]))

    with caplog.at_level(logging.WARNING):
        assert manager.load_json(path) is None

    assert any("not an object" in r.getMessage() for r in caplog.records)


def test_empty_yaml_is_distinguishable_from_a_broken_one(manager, tmp_path, caplog):
    path = _write(tmp_path, "empty.yaml", "")

    with caplog.at_level(logging.WARNING):
        assert manager.load_yaml(path) is None

    assert any("empty" in r.getMessage().lower() for r in caplog.records)


def test_missing_file_returns_none_without_raising(manager, tmp_path):
    assert manager.load_yaml(tmp_path / "nope.yaml") is None
    assert manager.load_json(tmp_path / "nope.json") is None


def test_valid_documents_round_trip(manager, tmp_path):
    yaml_path = _write(tmp_path, "good.yaml", "a: 1\nb:\n  c: 2\n")
    json_path = _write(tmp_path, "good.json", json.dumps({"a": 1, "b": {"c": 2}}))

    assert manager.load_yaml(yaml_path) == {"a": 1, "b": {"c": 2}}
    assert manager.load_json(json_path) == {"a": 1, "b": {"c": 2}}


def test_save_then_load_is_stable(manager, tmp_path):
    payload = {"nested": {"values": [1, 2, 3]}, "flag": True}

    manager.save_yaml(payload, tmp_path / "rt.yaml")
    manager.save_json(payload, tmp_path / "rt.json")

    assert manager.load_yaml(tmp_path / "rt.yaml") == payload
    assert manager.load_json(tmp_path / "rt.json") == payload
