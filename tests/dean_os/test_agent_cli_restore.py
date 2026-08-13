"""The prose docs must not advertise commands that do not exist.

This used to assert that every ``run_agent_*.py`` named in the prose docs existed
on disk. That assertion was correct in spirit but useless in practice: it had been
failing with a 93-item blob, so it reported "the docs drifted" without saying which
gap was new and which was a known decision. A blob that always fails teaches
everyone to ignore it.

Now the rule is: a command named in the prose docs must either exist as a file, or
be recorded in ``dean_os/config/retired_commands.yaml`` with a reason. A newly
invented command fails immediately, while known gaps stay visible and explained.

``dean_os/COMMAND_INDEX.md`` is generated from the wrappers themselves and checked
for staleness here, so the machine-readable half of the docs cannot drift at all.
"""
from __future__ import annotations

import py_compile
from pathlib import Path

from dean_os.command_index import (
    build_command_index,
    documented_commands,
    load_retired,
    render_markdown,
    undocumented_drift,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_prose_docs_name_only_commands_that_exist_or_are_recorded_as_retired() -> None:
    documented = documented_commands(PROJECT_ROOT)
    assert "run_agent_tuning.py" in documented, "sanity check: the docs should mention run_agent_tuning.py"

    drift = undocumented_drift(PROJECT_ROOT)
    assert drift == {}, (
        "These commands are named in the prose docs but neither exist on disk nor appear in "
        "dean_os/config/retired_commands.yaml. Either add the wrapper, or record why it is "
        f"missing: {sorted(drift)}"
    )


def test_retired_registry_has_no_entries_that_actually_exist() -> None:
    """Closing a gap means deleting its line, so a stale entry is itself drift."""
    retired = load_retired()
    resurrected = sorted(name for name in retired if (PROJECT_ROOT / name).exists())
    assert resurrected == [], (
        "These commands now exist on disk, so their entries in "
        f"dean_os/config/retired_commands.yaml should be deleted: {resurrected}"
    )


def test_every_retired_entry_carries_a_reason() -> None:
    retired = load_retired()
    assert retired, "retired_commands.yaml should not be empty while gaps remain"
    empty = sorted(name for name, reason in retired.items() if len(reason.strip()) < 20)
    assert empty == [], f"Retired commands need an explanation, not a placeholder: {empty}"


def test_command_index_is_not_stale() -> None:
    index_path = PROJECT_ROOT / "dean_os" / "COMMAND_INDEX.md"
    assert index_path.exists(), "Run: python run_agent_command_index.py"

    expected = render_markdown(build_command_index(PROJECT_ROOT), load_retired())
    assert index_path.read_text(encoding="utf-8") == expected, (
        "dean_os/COMMAND_INDEX.md is out of date. Regenerate it with: "
        "python run_agent_command_index.py"
    )


def test_command_index_covers_every_wrapper_on_disk() -> None:
    wrappers = {path.name for path in PROJECT_ROOT.glob("run_agent_*.py")}
    index = build_command_index(PROJECT_ROOT)
    assert set(index) == wrappers
    assert len(wrappers) >= 24


def test_no_wrapper_failed_to_parse() -> None:
    index = build_command_index(PROJECT_ROOT)
    broken = {name: info["parse_error"] for name, info in index.items() if info.get("parse_error")}
    assert broken == {}, f"Wrappers that could not be parsed: {broken}"


def test_every_wrapper_declares_a_description() -> None:
    index = build_command_index(PROJECT_ROOT)
    undescribed = sorted(name for name, info in index.items() if not info.get("description"))
    assert undescribed == [], (
        "Every wrapper needs an ArgumentParser(description=...) so the generated index "
        f"can say what it does: {undescribed}"
    )


def test_run_agent_wrappers_compile() -> None:
    wrappers = sorted(PROJECT_ROOT.glob("run_agent_*.py"))
    assert len(wrappers) >= 24
    for wrapper in wrappers:
        py_compile.compile(str(wrapper), doraise=True)
