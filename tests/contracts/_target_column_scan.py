"""Detect feature code that CREATES a `target_*` column.

The original contract checks simply asked whether the substring "target_"
appeared anywhere in a file. That flagged, among others:

  - `src/features/enrichers/context_map_enricher.py`, whose only match is
    `[c for c in context_columns if not c.startswith('target_')]` -- the very
    line that ENFORCES the rule being tested;
  - `src/features/enrichers/derived_features_enricher.py`, where every match
    is the parameter `target_column` meaning "which price column to derive
    from", defaulting to 'close', and whose outputs are LAG_*, VELOCITY_* and
    ACCELERATION_*;
  - `src/features/validation/feature_leakage_guard.py`, a module whose entire
    purpose is preventing target leakage;
  - the feature selectors, which must reference the target to score relevance.

A check that fires on its own guard code is noise, and noise teaches people to
silence the guard. These patterns look for the actual failure mode instead:
assigning into a frame under a `target_`-prefixed column name.
"""
from __future__ import annotations

import re
from pathlib import Path

#: Direct creation of a target column, e.g.
#:   df['target_up_1d'] = ...      df[f"target_{name}"] = ...
_SUBSCRIPT_ASSIGN = re.compile(
    r"""\[\s*f?["']target_[^"']*["']\s*\]\s*=(?!=)""",
    re.VERBOSE,
)

#: DataFrame.assign(target_x=...)
_ASSIGN_KWARG = re.compile(r"\.assign\(\s*target_\w+\s*=")

#: Renaming an existing column INTO a target name, e.g.
#:   .rename(columns={"close": "target_close"})
_RENAME_INTO = re.compile(r"""rename\([^)]*["']target_\w+["']\s*\}""", re.DOTALL)

PATTERNS = (
    ("subscript assignment", _SUBSCRIPT_ASSIGN),
    ("assign() keyword", _ASSIGN_KWARG),
    ("rename into target name", _RENAME_INTO),
)

AUDIT_IGNORE = "audit-ignore: TARGET_IN_FEATURE_MODULE"


def target_column_creations(text: str) -> list[str]:
    """Return descriptions of every apparent target-column creation in `text`."""
    hits: list[str] = []
    for label, pattern in PATTERNS:
        for match in pattern.finditer(text):
            hits.append(f"{label}: {match.group(0).strip()[:80]}")
    return hits


def scan_tree(root: Path, *, honour_audit_ignore: bool = True) -> dict[str, list[str]]:
    """Map file path -> target-column creations found in it."""
    offenders: dict[str, list[str]] = {}
    if not root.exists():
        return offenders
    for path in root.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if honour_audit_ignore and AUDIT_IGNORE in text:
            continue
        hits = target_column_creations(text)
        if hits:
            offenders[str(path)] = hits
    return offenders
