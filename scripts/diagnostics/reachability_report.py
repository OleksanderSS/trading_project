"""Which modules can actually be reached, and which only look like they can.

Run this before any cleanup pass. It exists because the naive version of this
question -- "who imports X?" -- gave a confidently wrong answer on 2026-08-12:
98 modules with no importer, 13,250 lines, and most of that list was alive.

Four ways a live module has no importer in this repository:

  DISCOVERED    `FeatureOrchestrator.create_from_config` imports every module
                in `src/features/enrichers/` and keeps the BaseEnricher
                subclasses whose id is true in `enabled_enrichers`. All 17
                working enrichers are "unimported". So is the config-gated
                collector factory.
  ROOT          The import scan used to walk src/tests/scripts and skip the
                repository root -- where run_hybrid_pipeline.py lives. The
                largest finding of that day was on line 30 of it.
  REEXPORT      A module reached only through its package's __init__ is live
                if anything imports the package attribute.
  ENTRY POINT   A script run as `python path.py` is imported by nobody by
                definition.

And one way a module has importers and is dead anyway:

  ARCHIVE-ONLY  every importer lives under an archive/ directory. That is how
                `src/main/modes/` survived: its dispatcher, system_orchestrator,
                was archived and the six modes it dispatched were left behind,
                each still importing modes/base.py and each other.

So this reports buckets, not a delete list. Every candidate still needs the
question the buckets cannot answer: does it run? The cheapest ground truth is
logs/ -- a module that runs says so.

    python scripts/diagnostics/reachability_report.py [--verbose]
"""
from __future__ import annotations

import argparse
import ast
import collections
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

SCAN_ROOTS = ("src", "tests", "scripts")
SKIP_PARTS = {"archive", "__pycache__", "draft", "dead_pipeline_code"}

# Packages whose members are imported by a walk rather than by name. A module
# here is reachable by construction; whether it RUNS is a config question.
DISCOVERED_PACKAGES = {
    "src/features/enrichers": (
        "walked by FeatureOrchestrator._discover_enrichers_in_module; "
        "gated by features.enabled_enrichers"
    ),
    "src/data/collectors": (
        "instantiated by CollectorFactory from the collectors config; "
        "gated by collectors.<name>.enabled"
    ),
}


def _python_files(root: Path, *, skip_archived: bool) -> list[Path]:
    blocked = SKIP_PARTS if skip_archived else SKIP_PARTS - {"archive"}
    return [
        path
        for path in sorted(root.rglob("*.py"))
        if not blocked & set(path.parts)
    ]


def _collect_files(*, skip_archived: bool = True) -> list[Path]:
    files: list[Path] = []
    for name in SCAN_ROOTS:
        directory = PROJECT_ROOT / name
        if directory.is_dir():
            files.extend(_python_files(directory, skip_archived=skip_archived))
    # The repository root itself: entry points live here and the scan that
    # missed them missed the biggest finding of the pass that wrote this file.
    files.extend(sorted(PROJECT_ROOT.glob("*.py")))
    return files


def _module_name(path: Path) -> str:
    relative = path.relative_to(PROJECT_ROOT)
    return ".".join(relative.with_suffix("").parts).replace(".__init__", "")


def _build_import_graph(files: list[Path]) -> dict[str, set[Path]]:
    """target module name -> files importing it (module and symbol forms)."""
    importers: dict[str, set[Path]] = collections.defaultdict(set)
    for path in files:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    importers[alias.name].add(path)
            elif isinstance(node, ast.ImportFrom):
                target = _resolve_import_from(node, path)
                if not target:
                    continue
                importers[target].add(path)
                for alias in node.names:
                    importers[f"{target}.{alias.name}"].add(path)
    return importers


def _resolve_import_from(node: ast.ImportFrom, path: Path) -> str | None:
    """Turn `from .evaluation.orchestrator import X` into its absolute name.

    Relative imports were a blind spot in the first version of this script,
    and an expensive one: `src/pipeline/stages/stage_7_evaluation.py` is a
    fourteen-line facade that subclasses the 699-line
    `evaluation/orchestrator.py` through `from .evaluation.orchestrator
    import EvaluationStage`. Recorded verbatim, that key never matches the
    absolute module name, so the largest live module in Stage 7 was reported
    as UNREFERENCED. Every relative import in the repository was invisible
    the same way.
    """
    if node.level == 0:
        return node.module
    package_parts = list(path.relative_to(PROJECT_ROOT).with_suffix("").parts)
    if path.name == "__init__.py":
        package_parts = package_parts[:-1]
    # level 1 is the containing package, level 2 its parent, and so on.
    base = package_parts[: len(package_parts) - node.level]
    if not base:
        return node.module
    return ".".join(base + ([node.module] if node.module else []))


def _discovered_reason(path: Path) -> str | None:
    posix = path.relative_to(PROJECT_ROOT).as_posix()
    for package, reason in DISCOVERED_PACKAGES.items():
        if posix.startswith(f"{package}/") and path.name != "__init__.py":
            return reason
    return None


def _is_archived(path: Path) -> bool:
    return "archive" in path.parts


def classify(verbose: bool = False) -> dict[str, list[tuple[str, str]]]:
    files = _collect_files()
    # Archived code is never CLASSIFIED, but it must be PARSED: a module whose
    # only importers are archived looks live to a graph that never read them.
    # That is how src/main/modes/ hid -- its dispatcher was archived and the
    # six modes it dispatched kept importing each other.
    importers = _build_import_graph(_collect_files(skip_archived=False))
    buckets: dict[str, list[tuple[str, str]]] = collections.defaultdict(list)

    for path in files:
        if not path.is_relative_to(PROJECT_ROOT / "src"):
            continue
        if path.name == "__init__.py":
            continue

        name = _module_name(path)
        reason = _discovered_reason(path)
        if reason:
            buckets["DISCOVERED"].append((name, reason))
            continue

        found = {other for other in importers.get(name, ()) if other != path}
        if not found:
            buckets["UNREFERENCED"].append((name, "no import of this module anywhere"))
            continue

        live = {other for other in found if not _is_archived(other)}
        if not live:
            witness = sorted(found)[0].relative_to(PROJECT_ROOT).as_posix()
            buckets["ARCHIVE_ONLY"].append(
                (name, f"every importer is archived, e.g. {witness}")
            )
            continue

        direct = {other for other in live if other.name != "__init__.py"}
        if not direct:
            witness = sorted(live)[0].relative_to(PROJECT_ROOT).as_posix()
            buckets["REEXPORT_ONLY"].append((name, f"reached only via {witness}"))
            continue

        if verbose:
            buckets["LIVE"].append((name, f"{len(direct)} importer(s)"))

    return buckets


ORDER = ("UNREFERENCED", "ARCHIVE_ONLY", "REEXPORT_ONLY", "DISCOVERED", "LIVE")

HEADINGS = {
    "UNREFERENCED": "Nothing imports these. Check logs/ before believing it.",
    "ARCHIVE_ONLY": "Reachable only from archived code -- orphaned subsystems.",
    "REEXPORT_ONLY": "Reached only through a package __init__.",
    "DISCOVERED": "Loaded by a walk or a factory; ask the config, not the graph.",
    "LIVE": "Imported directly by live code.",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbose", action="store_true", help="also list live modules")
    args = parser.parse_args()

    buckets = classify(verbose=args.verbose)

    for bucket in ORDER:
        rows = buckets.get(bucket)
        if not rows:
            continue
        print(f"\n=== {bucket} ({len(rows)}) — {HEADINGS[bucket]}")
        for name, reason in sorted(rows):
            print(f"  {name:70s} {reason}")

    print(
        "\nNone of these buckets is a delete list. UNREFERENCED and "
        "ARCHIVE_ONLY are where to look first; confirm against logs/ that the "
        "module really never runs, and check whether its capability is already "
        "provided elsewhere before removing anything."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
