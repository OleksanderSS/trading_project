"""Find operations that assume the rows are in time order.

`ffill`, `shift`, `rolling`, `diff`, `cumsum` and `expanding` all walk ROWS.
On a frame holding every ticker stacked, and in whatever order an earlier step
left it, that is not the same as walking time -- and the difference is
lookahead, not untidiness.

Two were found by hand on 2026-08-28, hours apart:

  * the macro fill put the 2024 CPI on every row from 1996 to 2023, because a
    newest-first frame turns ffill into a thirty-year backfill;
  * the champion regime spread along row order, leaving 16 of 110 tickers with
    the same state on every row of their history.

Neither is visible by reading: the calls look correct, and there is no `bfill`
anywhere in the codebase. The question they fail is not "what does this call
do" but "in what order are the rows".

This lists every such call in the enrichment path together with whether its
enclosing function sorts by time or groups by ticker first. A call with
neither is a candidate, not a verdict -- some operate on a single name's frame
where the order is guaranteed by the caller.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

#: Everything that walks rows and would change meaning if they were reordered.
ORDER_DEPENDENT = {
    "ffill", "bfill", "pad", "backfill",
    "shift", "diff", "cumsum", "cumprod", "cummax", "cummin",
    "rolling", "expanding", "ewm", "pct_change", "interpolate",
}

#: What makes such a call safe: an explicit ordering, or a per-name grouping.
GUARDS = {"sort_values", "sort_index", "groupby", "resample", "asof", "merge_asof"}

SEARCH = [
    "src/features/enrichers",
    "src/features/utils",
    "src/pipeline/stages/feature_engineering",
    "src/analytics/calculators",
]


def _calls(node: ast.AST) -> set[str]:
    names = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute):
            names.add(child.func.attr)
    return names


def scan() -> list[tuple[str, int, str, str, bool]]:
    findings = []
    for folder in SEARCH:
        for path in sorted((ROOT / folder).rglob("*.py")):
            if "__pycache__" in str(path) or "archive" in str(path):
                continue
            try:
                tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
            except SyntaxError:
                continue
            for func in ast.walk(tree):
                if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                used = _calls(func)
                risky = sorted(used & ORDER_DEPENDENT)
                if not risky:
                    continue
                guarded = bool(used & GUARDS)
                findings.append((
                    str(path.relative_to(ROOT)).replace("\\", "/"),
                    func.lineno, func.name, ",".join(risky), guarded,
                ))
    return findings


def main() -> int:
    findings = scan()
    unguarded = [f for f in findings if not f[4]]

    print(f"functions using row-order operations: {len(findings)}")
    print(f"  with an explicit sort or groupby: {len(findings) - len(unguarded)}")
    print(f"  WITHOUT either -- candidates:     {len(unguarded)}\n")

    for path, line, name, ops, _ in unguarded:
        print(f"  {path}:{line}")
        print(f"      {name}()  ->  {ops}")

    print(
        "\nA candidate is not a defect: some of these take a single name's "
        "frame,\nwhere the caller guarantees the order. What none of them do "
        "is say so."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
