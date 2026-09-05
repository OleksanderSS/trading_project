"""Which mechanisms are referenced by their own tests and by nothing else.

WHY, IN THREE ACCIDENTS ON ONE DAY. `apply_seal` -- the function
SEALED_HOLDOUT.md calls the enforcement mechanism -- had zero callers, so the
seal rested on every diagnostic remembering to filter (#264). `universe_as_of`
had zero callers with the store filled and eleven tests green, so the free half
of the survivorship fix measured nothing (Р46). `validation_predictions` was
produced per fold, unit-tested, and read by no one, so every out-of-sample
number came from one market episode (#47). Each was found by reading something
else. Three accidents is a shape.

THIS IS A READING LIST, NOT A VERDICT, and the reason is documented history:
commit dabe5540 archived `CriticalSignalDetector` for having "zero callers",
which was false -- analysis.yaml constructs it by module path. Static
reachability cannot see config-driven loading, factories or getattr. So names
that appear in any config are removed before anything is printed, and what
remains still has to be read.

WHAT IT WOULD AND WOULD NOT HAVE CAUGHT. `apply_seal` and `universe_as_of` are
public module-level names referenced only by tests: both would have appeared
here. `validation_predictions` is a dict KEY, which this cannot see -- two of
three, stated rather than implied.

MEASURED BEFORE BEING WRITTEN, because three scanner proposals were measured
and refused this week (WORKING_METHOD) and a fourth needed a better reason
than enthusiasm:

    naive pass                                     81 names
    minus config-constructed (the documented blind spot)
    minus names used inside their own module        16 names
    of those, modules that SAY nothing calls them    4
    of those, nothing says it was on purpose        12

Hand-checked eight of the twelve against the live tree: eight of eight were
genuinely unreferenced, every apparent caller being in `src/archive/`. That
precision is why this one was built and the other three were not -- their
rules scored 0 of 5 and 0 of 47.

AND THE HONEST RESULT OF THE FIRST RUN: none of the twelve is another
`apply_seal`. They are unused helpers, an unused wrapper, and two halves of an
API whose other half is wired. The value here is preventive, not a haul.

    python scripts/diagnostics/what_has_no_caller.py
    python scripts/diagnostics/what_has_no_caller.py --all
"""
from __future__ import annotations

import argparse
import ast
import io
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

#: Pruned in the loop, with a deadline. An unpruned tree walk hung twice in
#: this project, the second time for forty minutes of the owner's evening.
SKIP_DIRS = {"__pycache__", ".git", "node_modules", "archive", "dean_os",
             "data", "diagnostic_reports", ".venv", "venv", "logs", "mlruns"}
TIME_BUDGET = 180

#: Phrases a module uses to say "nothing calls this, and that is known".
#:
#: The difference this whole diagnostic turns on. `stages/modeling/utils.py`
#: opens with "despite the name, ModelingStage does NOT call these ... Kept
#: live rather than archived because they're real, tested, working
#: implementations". That is a decision, recorded where a reader meets it.
#: `apply_seal` had no such note and simply looked wired. Same invariant as
#: `test_a_default_says_it_was_a_default`: silence must carry a mark.
MARKED = ("does not call", "does NOT call", "not called", "no caller",
          "zero callers", "currently unused", "unused by", "only exercised by",
          "never adopted", "not wired", "kept live rather than archived")


def _walk(*roots: str, suffixes: tuple[str, ...] = (".py",)):
    deadline = time.time() + TIME_BUDGET
    for top in roots:
        if not os.path.isdir(top):
            continue
        for dirpath, dirs, files in os.walk(top):
            if time.time() > deadline:
                raise SystemExit(
                    "the walk exceeded its budget; prune SKIP_DIRS further "
                    "rather than waiting"
                )
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
            for name in files:
                if name.endswith(suffixes):
                    yield os.path.join(dirpath, name)


def _public_names(path: str) -> list[tuple[str, str]]:
    """Top-level public functions, classes and CONSTANTS in one module."""
    try:
        tree = ast.parse(io.open(path, encoding="utf-8",
                                 errors="replace").read())
    except SyntaxError:
        return []
    found: list[tuple[str, str]] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not node.name.startswith("_"):
                found.append((node.name, "function"))
        elif isinstance(node, ast.ClassDef):
            if not node.name.startswith("_"):
                found.append((node.name, "class"))
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id.isupper() \
                        and not target.id.startswith("_"):
                    found.append((target.id, "constant"))
    return found


def scan() -> tuple[list[tuple], list[tuple], int]:
    """(unmarked, marked, removed_as_config_constructed)."""
    src_files = list(_walk("src"))
    production = list(_walk("src", "scripts")) + [
        name for name in os.listdir(".") if name.endswith(".py")
    ]
    tests = list(_walk("tests"))
    configs = list(_walk("src/config", "config", ".github",
                         suffixes=(".yaml", ".yml", ".json", ".toml")))

    read = lambda path: io.open(path, encoding="utf-8", errors="replace").read()
    prod_text = {path: read(path) for path in production}
    test_text = "\n".join(read(path) for path in tests)
    config_text = "\n".join(read(path) for path in configs)

    marked, unmarked, by_config = [], [], 0
    for path in src_files:
        text = prod_text.get(path, "")
        module_says = any(phrase in text for phrase in MARKED)
        for name, kind in _public_names(path):
            elsewhere = sum(other_text.count(name)
                            for other, other_text in prod_text.items()
                            if other != path)
            # Its OWN module counts, minus the definition line. Excluding the
            # defining file wholesale flagged SEAL_SHARE and NOT_MEASURED
            # while they were being used three lines below where they are
            # defined -- a different question from the one that matters.
            own = text.count(name) - 1
            if elsewhere or own > 0 or not test_text.count(name):
                continue
            if name in config_text:
                by_config += 1
                continue
            row = (kind, name, path.replace("\\", "/"), test_text.count(name))
            (marked if module_says else unmarked).append(row)
    return sorted(unmarked), sorted(marked), by_config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true",
                        help="also list the ones whose module says so")
    args = parser.parse_args()

    unmarked, marked, by_config = scan()

    print(f"removed as config-constructed (the documented blind spot): "
          f"{by_config}")
    print(f"no live caller, and the module SAYS SO: {len(marked)}")
    print(f"no live caller, and NOTHING says that was on purpose: "
          f"{len(unmarked)}\n")

    print(f"{'kind':<10}{'name':<36}{'module'}")
    print("-" * 92)
    for kind, name, path, _ in unmarked:
        print(f"{kind:<10}{name:<36}{path}")

    if args.all and marked:
        print(f"\nDECLARED UNUSED, which is a decision and not a finding:")
        for kind, name, path, _ in marked:
            print(f"{kind:<10}{name:<36}{path}")

    print("\nNot proof. Config-driven construction, factories and getattr are "
          "invisible here\n(commit dabe5540 archived a LIVE analyzer for "
          "'zero callers'), and a dict key\ncannot be seen at all -- which is "
          "why #47 needed reading rather than scanning.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
