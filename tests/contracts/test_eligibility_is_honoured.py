"""
The eligibility flag must be read, not just written.

Three collectors set eligible_for_training=False on rows they cannot vouch
for. For a long time nothing outside those collectors read it, which made
test_sample_fallback_requires_opt_in_by_source_scan's green a false comfort:
the fabricated rows were gated in name only.
"""
from __future__ import annotations

import ast
from pathlib import Path

SRC = Path("src")
FLAG = "eligible_for_training"


def _modules():
    for path in sorted(SRC.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        yield path, path.read_text(encoding="utf-8", errors="ignore")


def test_something_outside_the_collectors_reads_the_flag():
    readers = [
        str(path)
        for path, text in _modules()
        if FLAG in text and "collectors" not in path.parts
    ]
    assert readers, (
        f"No module outside src/data/collectors mentions {FLAG}. The flag is "
        "written by collectors and read by nobody, so ineligible rows reach "
        "training anyway."
    )


def test_table_reads_filter_by_eligibility_by_default():
    """
    DataManager.fetch_data_from_table is the choke point every table read goes
    through. Its eligible_only parameter must default to True — a filter that
    defaults to off is a filter nobody applies.
    """
    tree = ast.parse((SRC / "data/management/data_manager.py").read_text(encoding="utf-8"))

    defaults = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "fetch_data_from_table":
            args = node.args
            names = [a.arg for a in args.args]
            assert "eligible_only" in names, (
                "fetch_data_from_table must accept eligible_only so callers "
                "that genuinely need raw rows can opt out explicitly."
            )
            offset = len(names) - len(args.defaults)
            default = args.defaults[names.index("eligible_only") - offset]
            defaults.append(default)

    assert defaults, "fetch_data_from_table not found in data_manager.py"
    for default in defaults:
        assert isinstance(default, ast.Constant) and default.value is True, (
            "eligible_only must default to True: rows are marked ineligible "
            "precisely because the source could not vouch for them."
        )
