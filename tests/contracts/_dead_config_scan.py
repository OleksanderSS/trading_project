"""Find config keys that are read, sometimes logged, and never used.

A dead config key is worse than a missing one. A missing key fails loudly the
first time something looks for it. A dead key sits in the YAML looking like a
control, gets read into an attribute, often gets printed in a startup line --
and decides nothing. Editing it changes the log and nothing else.

Three of these have cost real time here:

  `daily_max_years: 2`  sat in the yahoo_finance block with zero readers and
                        read like the live limit. Cost an afternoon, and the
                        note left behind says "removed rather than wired,
                        because two places declaring one number is how every
                        fix in this project has half-landed".
  `vix.params.period`   was read into `self.period`, printed in
                        "VIXCollector initialized. Period: 30d", and then the
                        fetch said `history(period="60d")` outright. The
                        config declared 30, sixty were collected, and neither
                        number had been chosen by anyone.
  `attention_window`    a 20 in `TickerExternalEnricher.__init__` that no
                        config file set and no line read.

The scan runs in two passes because one pass over-reports badly.

  Pass 1 finds `self.x = <something>.get(...)` where the source mentions
  config/param/setting, then checks every load of `self.x` INSIDE the class.
  This alone flags `table_name` and `timeout` on half the collectors, because
  those are read by the BASE class, not the subclass that assigns them.

  Pass 2 therefore looks for `.x` across the whole of `src/`, skipping the
  assignment itself and any line that logs. Only attributes with no reader
  anywhere survive. On 2026-08-22 that took 39 candidates down to 22.

Loads inside logging calls do not count as use. That is the whole point: the
VIX period was "used" in exactly that sense.

Runnable standalone:  python tests/contracts/_dead_config_scan.py
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SRC_ROOT = PROJECT_ROOT / "src"

EXCLUDED_PARTS = ("archive", "draft", "__pycache__", ".venv", "venv")

#: Names that mark a call as logging rather than use.
_LOGGING_HINTS = ("logger", "log", "print", "debug", "info", "warning",
                  "error", "exception", "critical")

#: What the object being `.get()` from has to look like for the read to count
#: as a configuration read.
_CONFIG_HINTS = ("config", "param", "setting", "opts", "options")


@dataclass(frozen=True)
class Finding:
    path: str
    cls: str
    attribute: str
    line: int
    logged: int

    def __str__(self) -> str:
        where = f"logged {self.logged}x" if self.logged else "never read at all"
        return f"{self.path}:{self.line} {self.cls}.{self.attribute} [{where}]"


def _config_read(node: ast.AST) -> bool:
    """True when the expression pulls a value out of a config-like mapping."""
    for sub in ast.walk(node):
        if (isinstance(sub, ast.Call)
                and isinstance(sub.func, ast.Attribute)
                and sub.func.attr == "get"):
            source = ast.unparse(sub.func.value).lower()
            if any(hint in source for hint in _CONFIG_HINTS):
                return True
    return False


def _parents(tree: ast.AST) -> dict[ast.AST, ast.AST]:
    mapping: dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            mapping[child] = parent
    return mapping


def _inside_logging(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> bool:
    current = parents.get(node)
    while current is not None:
        if isinstance(current, ast.Call) and hasattr(current, "func"):
            text = ast.unparse(current.func).lower()
            if any(hint in text for hint in _LOGGING_HINTS):
                return True
        current = parents.get(current)
    return False


@lru_cache(maxsize=1)
def _sources() -> tuple[tuple[str, str], ...]:
    collected = []
    for path in sorted(SRC_ROOT.rglob("*.py")):
        if any(part in EXCLUDED_PARTS for part in path.parts):
            continue
        collected.append((str(path), path.read_text(encoding="utf-8", errors="replace")))
    return tuple(collected)


def _read_anywhere(attribute: str) -> bool:
    """True when `.attribute` is read somewhere that is not an assignment or a log."""
    pattern = re.compile(rf"\.{re.escape(attribute)}\b")
    assignment = re.compile(rf"self\.{re.escape(attribute)}\s*=")
    for _, text in _sources():
        for match in pattern.finditer(text):
            start = text.rfind("\n", 0, match.start()) + 1
            end = text.find("\n", match.end())
            line = text[start: end if end > 0 else len(text)]
            if assignment.search(line):
                continue
            if any(hint in line.lower() for hint in ("logger", "log.", "print(")):
                continue
            return True
    return False


def scan() -> list[Finding]:
    """Config-derived attributes with no reader anywhere in src/."""
    findings: list[Finding] = []
    for path, text in _sources():
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        parents = _parents(tree)
        relative = Path(path).relative_to(PROJECT_ROOT).as_posix()

        for cls in ast.walk(tree):
            if not isinstance(cls, ast.ClassDef):
                continue

            assigned: dict[str, int] = {}
            for node in ast.walk(cls):
                if not (isinstance(node, ast.Assign) and len(node.targets) == 1):
                    continue
                target = node.targets[0]
                if not (isinstance(target, ast.Attribute)
                        and isinstance(target.value, ast.Name)
                        and target.value.id == "self"):
                    continue
                if _config_read(node.value):
                    assigned.setdefault(target.attr, node.lineno)

            if not assigned:
                continue

            used = dict.fromkeys(assigned, 0)
            logged = dict.fromkeys(assigned, 0)
            for node in ast.walk(cls):
                if not (isinstance(node, ast.Attribute)
                        and isinstance(node.ctx, ast.Load)
                        and isinstance(node.value, ast.Name)
                        and node.value.id == "self"
                        and node.attr in assigned):
                    continue
                if _inside_logging(node, parents):
                    logged[node.attr] += 1
                else:
                    used[node.attr] += 1

            for attribute, line in assigned.items():
                if used[attribute] == 0 and not _read_anywhere(attribute):
                    findings.append(
                        Finding(relative, cls.name, attribute, line, logged[attribute])
                    )
    return sorted(findings, key=lambda finding: (finding.path, finding.line))


if __name__ == "__main__":
    results = scan()
    for finding in results:
        print(finding)
    print(f"\n{len(results)} config keys that decide nothing")
