from __future__ import annotations

import py_compile
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOC_PATHS = [
    PROJECT_ROOT / "dean_os" / "COMMAND_CHECKLIST.md",
    PROJECT_ROOT / "dean_os" / "IMPLEMENTATION_STATUS.md",
    PROJECT_ROOT / "dean_os" / "NEXT_CHAT_HANDOFF.md",
    PROJECT_ROOT / "Agents_architecture.md",
]


def _documented_run_agent_wrappers() -> set[str]:
    wrappers: set[str] = set()
    for path in DOC_PATHS:
        if not path.exists():
            continue
        wrappers.update(re.findall(r"run_agent_[A-Za-z0-9_]+\.py", path.read_text(encoding="utf-8", errors="replace")))
    return wrappers


def test_documented_run_agent_wrappers_exist() -> None:
    wrappers = _documented_run_agent_wrappers()
    assert "run_agent_tuning.py" in wrappers
    missing = sorted(wrapper for wrapper in wrappers if not (PROJECT_ROOT / wrapper).exists())
    assert missing == []


def test_run_agent_wrappers_compile() -> None:
    wrappers = sorted(PROJECT_ROOT.glob("run_agent_*.py"))
    assert len(wrappers) >= 24
    for wrapper in wrappers:
        py_compile.compile(str(wrapper), doraise=True)

