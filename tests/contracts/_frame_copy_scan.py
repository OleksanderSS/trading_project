"""Find frames copied deeply when only whole columns are written.

`df.copy()` duplicates every data block. `df.copy(deep=False)` duplicates the
column index and shares the blocks. When a function only ever assigns WHOLE
columns into the copy, the two behave identically -- and the deep one pays for
the entire frame.

This is not a micro-optimisation. The defect killed three separate runs:

  #23  `_initial_feature_columns` copied the stage-3 frame to read its column
       names. Fixed 2026-08-19.
  #84  `_select_features` did the same thing a few hundred lines below, and
       went on allocating after #23 was closed: "Unable to allocate 4.15 GiB
       for an array with shape (2151, 259133)".
  #95  `ensure_datetime_column` copied the frame to normalise one column and
       killed rebuild v6 with "Unable to allocate 4.25 GiB".

Three instances of one shape, each found only when a run died. Measured on a
2200 x 5000 stand-in and scaled to the real stage-3 frame -- 259,133 ROWS by
2,238 columns, confirmed against the batch -- one such copy costs ~4.25 GiB
and ~4.9 s. (Numpy prints the failing BLOCK, which is transposed, so the
tracebacks read `shape (2200, 259133)`: columns first, then rows. The bytes
are the same either way.)

The scan is deliberately conservative. A function is reported ONLY when the
deep copy is provably unnecessary:

  * the copied name is a parameter of the function -- so the frame belongs to
    the caller and may be arbitrarily large;
  * every write into the copy is a whole-column assignment (`result[col] = ...`);
  * there is no partial write (`.loc[...] =`, `.iloc[...] =`, `.at`, `.values`)
    and no `inplace=True` call.

That last condition is the one that matters. A partial write into a shallow
copy DOES leak back into the original -- verified on pandas 2.3.3, where
`s = o.copy(deep=False); s.loc[0, "a"] = 99` changes `o`. Whole-column
assignment, new or existing, does not. Anything with a partial write is left
alone, because there the deep copy is load-bearing.

Runnable standalone:  python tests/contracts/_frame_copy_scan.py
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SCAN_ROOTS = ("src", "dean_os")

EXCLUDED_PARTS = ("archive", "draft", "__pycache__", ".venv", "venv")

#: Attribute accessors whose subscript assignment writes INTO existing data
#: rather than rebinding a whole column. These make a shallow copy unsafe.
_PARTIAL_WRITE_ACCESSORS = ("loc", "iloc", "at", "iat", "values")

#: Parameter names that plausibly hold a DataFrame. Checked against the name
#: only -- the codebase is not annotated consistently enough to rely on types.
_FRAME_HINTS = ("df", "frame", "data", "features", "prices", "bars", "signals")


@dataclass(frozen=True)
class Finding:
    path: str
    function: str
    line: int
    parameter: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line} {self.function}({self.parameter})"


def _looks_like_a_frame(name: str) -> bool:
    lowered = name.lower()
    return any(hint in lowered for hint in _FRAME_HINTS)


def _writes_partially(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """True when the function writes into a slice rather than a whole column."""
    for node in ast.walk(fn):
        if isinstance(node, ast.Assign | ast.AugAssign | ast.AnnAssign):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if not isinstance(target, ast.Subscript):
                    continue
                base = target.value
                if (isinstance(base, ast.Attribute)
                        and base.attr in _PARTIAL_WRITE_ACCESSORS):
                    return True
        if isinstance(node, ast.Call):
            for keyword in node.keywords:
                if keyword.arg == "inplace" and getattr(keyword.value, "value", None) is True:
                    return True
    return False


def _scan_function(
    fn: ast.FunctionDef | ast.AsyncFunctionDef,
    relative: str,
) -> list[Finding]:
    parameters = {
        argument.arg
        for argument in list(fn.args.args) + list(fn.args.kwonlyargs)
    }
    if not parameters or _writes_partially(fn):
        return []

    findings = []
    for node in ast.walk(fn):
        if not (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "copy"
                and isinstance(node.func.value, ast.Name)):
            continue
        if any(keyword.arg == "deep" for keyword in node.keywords):
            continue
        name = node.func.value.id
        if name not in parameters or not _looks_like_a_frame(name):
            continue
        findings.append(Finding(relative, fn.name, node.lineno, name))
    return findings


def scan(root: Path | None = None) -> list[Finding]:
    """Every provably-unnecessary deep frame copy, sorted by location."""
    base = root or PROJECT_ROOT
    findings: list[Finding] = []
    for directory in SCAN_ROOTS:
        source_root = base / directory
        if not source_root.exists():
            continue
        for path in sorted(source_root.rglob("*.py")):
            if any(part in EXCLUDED_PARTS for part in path.parts):
                continue
            try:
                tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
            except SyntaxError:
                # A file that does not parse cannot be reasoned about; the
                # syntax itself is another test's problem, not this one's.
                continue
            relative = path.relative_to(base).as_posix()
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                    findings.extend(_scan_function(node, relative))
    return sorted(findings, key=lambda finding: (finding.path, finding.line))


if __name__ == "__main__":
    results = scan()
    for finding in results:
        print(finding)
    print(f"\n{len(results)} deep copies of a caller's frame with no partial write")
