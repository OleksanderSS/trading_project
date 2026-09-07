"""
Shared AST helpers for the contract tests.

Source scans by substring were producing more noise than signal: scanning for
``target_`` flagged every ``target_column`` parameter, every ``target_series``
argument, and — most tellingly — ``feature_leakage_guard.py``, whose whole job
is to exclude ``target_*`` columns. A test that a module fails for mentioning
the thing it protects against cannot be acted on, so it gets ignored.

These helpers check the property the contracts actually mean: not "the file
contains this string" but "this module assigns a column with this name".
"""
from __future__ import annotations

import ast
from pathlib import Path


def _assigned_key(target: ast.AST) -> str | None:
    """
    Return the literal column name of a ``frame[<key>] = ...`` assignment.

    Handles plain string keys and f-strings, where the literal prefix is what
    matters (``df[f'target_{horizon}']`` still emits a target column).
    """
    if not isinstance(target, ast.Subscript):
        return None
    key = target.slice
    if isinstance(key, ast.Constant) and isinstance(key.value, str):
        return key.value
    if isinstance(key, ast.JoinedStr) and key.values:
        head = key.values[0]
        if isinstance(head, ast.Constant) and isinstance(head.value, str):
            return head.value
    return None


def columns_assigned(tree: ast.Module) -> list[tuple[int, str]]:
    """Every ``something[<literal>] = ...`` in the module, as (line, key)."""
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
            targets = [node.target]
        else:
            continue
        for target in targets:
            key = _assigned_key(target)
            if key is not None:
                found.append((node.lineno, key))
    return found


def modules_emitting_prefixed_columns(root: Path, prefix: str) -> list[str]:
    """Files under ``root`` that assign a column whose name starts with ``prefix``."""
    offenders: list[str] = []
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
        except SyntaxError as exc:
            offenders.append(f"{path}: does not parse: {exc}")
            continue
        hits = [f"{path}:{line}: {key}" for line, key in columns_assigned(tree)
                if key.startswith(prefix)]
        offenders.extend(hits)
    return offenders
