"""Find operations that read forward in time, in the source that performs them.

`TemporalLeakageGuard` used to look for `.shift(-1)` and `bfill` by matching
those strings against DataFrame COLUMN NAMES. A column is never called
"close.shift(-1)" -- those are Python expressions, so the check could not
fire. Measured on the 2026-08-02 export: 0 of 1,189 names contained "shift("
and 0 contained "bfill".

The check itself is worth having; it just has to run where the expression
exists. This scanner reads `src/` and reports:

  NEGATIVE_SHIFT    .shift(-n) -- reaches n bars into the future
  BACKFILL          bfill / method='backfill' -- fills a past gap with a
                    later value, which is the same leak wearing a different
                    hat and is far easier to write by accident
  CENTERED_WINDOW   rolling(..., center=True) -- a window at row i spans
                    i-n/2..i+n/2, so half of every result is computed from
                    bars that had not happened yet. The quietest of the
                    three: nothing in the expression says "future"

The third was added after the first two found src/processing/cleaners.py
clean_macro_data, which held a .bfill() AND two centred windows whose output
clipped the very rows they were computed from.

Both are legitimate when BUILDING A TARGET: a label is supposed to describe
what happens next. The project already marks those with
`# audit-ignore: NEGATIVE_SHIFT_LOOKAHEAD` / `NEGATIVE_SHIFT_INTENTIONAL`,
and this honours the same convention rather than inventing a second one.

Runnable standalone:  python tests/contracts/_lookahead_scan.py
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2] / "src"

#: Directories whose contents are not part of the live pipeline.
EXCLUDED_PARTS = ("archive", "draft", "__pycache__")

_NEGATIVE_SHIFT = re.compile(r"\.shift\s*\(\s*-\s*(?:\d+|[A-Za-z_])")
_BACKFILL = re.compile(
    r"\.bfill\s*\(|method\s*=\s*['\"](?:bfill|backfill)['\"]|\.fillna\s*\([^)]*backfill"
)
_CENTERED_WINDOW = re.compile(r"center\s*=\s*True")
#: The project's existing marker convention, honoured rather than replaced.
_IGNORE = re.compile(
    r"#\s*audit-ignore:\s*\S*(?:NEGATIVE_SHIFT|BACKFILL|LOOKAHEAD|CENTERED)"
)


@dataclass(frozen=True)
class Finding:
    path: str
    line: int
    kind: str
    text: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line}  [{self.kind}]  {self.text}"


def _code_only(source: str) -> dict[int, str]:
    """Line number -> that line with strings and comments blanked out.

    Prose is not an operation. A docstring explaining that .bfill() leaks, or
    a comment saying "do not use center=True", must not be reported as the
    thing it warns about -- this scanner reported its own documentation
    before this existed. Tokenising is exact where a "does the line start
    with #" heuristic is not: it also handles multi-line docstrings and a
    literal that merely mentions the text.
    """
    import io
    import tokenize

    lines = source.splitlines()
    blanked = {number: line for number, line in enumerate(lines, start=1)}
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        # An unparseable file still gets scanned raw rather than skipped: a
        # false positive is recoverable, a missed lookahead is not.
        return blanked

    for token in tokens:
        if token.type not in (tokenize.STRING, tokenize.COMMENT):
            continue
        start_row, end_row = token.start[0], token.end[0]
        for row in range(start_row, end_row + 1):
            if row not in blanked:
                continue
            if row == start_row and row == end_row:
                text = blanked[row]
                blanked[row] = (
                    text[: token.start[1]] + " " * (token.end[1] - token.start[1])
                    + text[token.end[1]:]
                )
            elif row == start_row:
                blanked[row] = blanked[row][: token.start[1]]
            elif row == end_row:
                blanked[row] = " " * token.end[1] + blanked[row][token.end[1]:]
            else:
                blanked[row] = ""
    return blanked


def scan(root: Path | None = None) -> list[Finding]:
    findings: list[Finding] = []
    base = root or SRC_ROOT
    for path in sorted(base.rglob("*.py")):
        if any(part in EXCLUDED_PARTS for part in path.parts):
            continue
        try:
            source = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        lines = source.splitlines()
        code = _code_only(source)
        relative = path.relative_to(base.parent).as_posix()
        for number, raw in enumerate(lines, start=1):
            # The ignore marker lives in a comment, so it is read from the
            # RAW line -- blanking removed it.
            if _IGNORE.search(raw):
                continue
            line = code.get(number, raw)
            stripped = raw.strip()
            if _NEGATIVE_SHIFT.search(line):
                findings.append(Finding(relative, number, "NEGATIVE_SHIFT", stripped[:120]))
            if _BACKFILL.search(line):
                findings.append(Finding(relative, number, "BACKFILL", stripped[:120]))
            if _CENTERED_WINDOW.search(line):
                findings.append(
                    Finding(relative, number, "CENTERED_WINDOW", stripped[:120])
                )
    return findings


def counts(root: Path | None = None) -> dict[str, int]:
    tally: dict[str, int] = {}
    for finding in scan(root):
        tally[finding.kind] = tally.get(finding.kind, 0) + 1
    return tally


if __name__ == "__main__":
    results = scan()
    for finding in results:
        print(finding)
    print(f"\n{len(results)} finding(s): {counts()}")
