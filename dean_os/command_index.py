"""Derive the DEAN-OS command index from the run_agent_*.py wrappers themselves.

Why this exists
---------------
``dean_os/COMMAND_CHECKLIST.md`` is hand-written prose that several different
agents edited over time. By 2026-08-13 it advertised 192 ``run_agent_*.py``
commands, of which 93 did not exist -- the docs had drifted from the code with
nothing to catch it. Restoring the missing commands by matching names turned out
to be actively dangerous: name similarity picks the wrong module, so a wrapper
named X ends up running Y.

So the command index is generated, not maintained. Reality is the wrappers on
disk; this module reads them and writes the index. A test regenerates it and
fails if the checked-in copy is stale, which makes that particular drift
impossible rather than merely detectable.

Parsing is static (``ast``): no wrapper is imported or executed, so building the
index cannot trigger a collector, a pipeline stage, or a network call, and it
stays fast enough to run inside the test suite.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INDEX_PATH = PROJECT_ROOT / "dean_os" / "COMMAND_INDEX.md"
RETIRED_PATH = PROJECT_ROOT / "dean_os" / "config" / "retired_commands.yaml"
WRAPPER_GLOB = "run_agent_*.py"

# Prose docs that reference commands and therefore need checking against reality.
PROSE_DOCS = [
    "dean_os/COMMAND_CHECKLIST.md",
    "dean_os/IMPLEMENTATION_STATUS.md",
    "dean_os/NEXT_CHAT_HANDOFF.md",
    "Agents_architecture.md",
]

GENERATED_HEADER = """<!-- GENERATED FILE -- DO NOT EDIT BY HAND.
     Regenerate with: python run_agent_command_index.py
     Source of truth: the run_agent_*.py wrappers in the project root. -->

# DEAN-OS Command Index

Every `run_agent_*.py` entrypoint in the project root, with the options each one
actually accepts. Generated from the wrappers themselves, so it cannot drift from
the code. For the reasoning and boundaries behind each workflow see
`dean_os/COMMAND_CHECKLIST.md`; for what a command is *for*, that prose is still
the place to look. This file only answers "does it exist, and what does it take".
"""


def _literal(node: ast.AST) -> Any:
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        return None


def _describe_default(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return value or '""'
    return repr(value)


def parse_wrapper(path: Path) -> dict[str, Any]:
    """Extract description and options from one wrapper, without importing it."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError, UnicodeDecodeError) as exc:
        return {"name": path.name, "parse_error": f"{type(exc).__name__}: {exc}", "options": []}

    description = None
    options: list[dict[str, Any]] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func

        # argparse.ArgumentParser(description=...) / ArgumentParser(description=...)
        name = getattr(func, "attr", None) or getattr(func, "id", None)
        if name == "ArgumentParser":
            for kw in node.keywords:
                if kw.arg == "description":
                    description = _literal(kw.value) or description
            continue

        # <parser>.add_argument("--flag", ...)
        if name != "add_argument":
            continue
        flags = [value for value in (_literal(arg) for arg in node.args) if isinstance(value, str)]
        if not flags:
            continue
        option: dict[str, Any] = {"flags": flags, "positional": not flags[0].startswith("-")}
        for kw in node.keywords:
            if kw.arg in {"default", "help", "action", "type", "nargs", "required", "dest"}:
                if kw.arg == "type":
                    option["type"] = getattr(kw.value, "id", None) or getattr(kw.value, "attr", None)
                else:
                    option[kw.arg] = _literal(kw.value)
        options.append(option)

    return {"name": path.name, "description": description, "options": options}


def build_command_index(project_root: str | Path = PROJECT_ROOT) -> dict[str, dict[str, Any]]:
    root = Path(project_root)
    return {path.name: parse_wrapper(path) for path in sorted(root.glob(WRAPPER_GLOB))}


def _render_option(option: dict[str, Any]) -> str:
    flags = ", ".join(f"`{flag}`" for flag in option["flags"])
    bits: list[str] = []
    if option.get("required"):
        bits.append("**required**")
    if option.get("action") in {"store_true", "store_false"}:
        bits.append("flag")
    elif option.get("type"):
        bits.append(str(option["type"]))
    elif option.get("action") == "append":
        bits.append("repeatable")
    if option.get("nargs"):
        bits.append(f"nargs={option['nargs']}")
    default = _describe_default(option.get("default"))
    if default is not None and not option.get("required"):
        bits.append(f"default `{default}`")
    suffix = f" — {', '.join(bits)}" if bits else ""
    help_text = option.get("help")
    if help_text:
        suffix += f". {help_text}"
    return f"- {flags}{suffix}"


def render_markdown(index: dict[str, dict[str, Any]], retired: dict[str, str] | None = None) -> str:
    lines = [GENERATED_HEADER, f"\n**{len(index)} commands.**\n"]

    for name, info in sorted(index.items()):
        lines.append(f"\n## `{name}`\n")
        if info.get("parse_error"):
            lines.append(f"Could not be parsed: {info['parse_error']}\n")
            continue
        if info.get("description"):
            lines.append(f"{info['description']}\n")
        options = info.get("options") or []
        if not options:
            lines.append("No command-line options.\n")
            continue
        positional = [item for item in options if item["positional"]]
        flagged = [item for item in options if not item["positional"]]
        if positional:
            lines.append("Positional:\n")
            lines.extend(_render_option(item) for item in positional)
            lines.append("")
        if flagged:
            lines.append("Options:\n")
            lines.extend(_render_option(item) for item in flagged)
            lines.append("")

    if retired:
        lines.append("\n## Retired / never implemented\n")
        lines.append(
            "Commands referenced by the prose docs that have no wrapper on disk. "
            "Listed here so a reference to a missing command is a recorded decision "
            "rather than silent drift.\n"
        )
        for name, reason in sorted(retired.items()):
            lines.append(f"- `{name}` — {reason}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def load_retired(path: str | Path = RETIRED_PATH) -> dict[str, str]:
    path = Path(path)
    if not path.exists():
        return {}
    import yaml

    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    commands = payload.get("retired_commands", payload)
    if not isinstance(commands, dict):
        return {}
    return {str(key): str(value) for key, value in commands.items()}


def documented_commands(project_root: str | Path = PROJECT_ROOT) -> dict[str, list[str]]:
    """Every run_agent_*.py name mentioned in the prose docs, with where."""
    root = Path(project_root)
    found: dict[str, list[str]] = {}
    for rel in PROSE_DOCS:
        path = root / rel
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for match in re.findall(r"run_agent_[A-Za-z0-9_]+\.py", text):
            found.setdefault(match, []).append(rel)
    return found


def undocumented_drift(project_root: str | Path = PROJECT_ROOT) -> dict[str, list[str]]:
    """Prose-doc commands that neither exist on disk nor are recorded as retired."""
    root = Path(project_root)
    retired = load_retired()
    return {
        name: docs
        for name, docs in documented_commands(root).items()
        if not (root / name).exists() and name not in retired
    }


def write_index(project_root: str | Path = PROJECT_ROOT) -> Path:
    root = Path(project_root)
    text = render_markdown(build_command_index(root), load_retired())
    target = root / "dean_os" / "COMMAND_INDEX.md"
    target.write_text(text, encoding="utf-8")
    return target


__all__ = [
    "INDEX_PATH",
    "build_command_index",
    "documented_commands",
    "load_retired",
    "parse_wrapper",
    "render_markdown",
    "undocumented_drift",
    "write_index",
]
