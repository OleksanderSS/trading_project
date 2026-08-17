"""Find joins and fills that carry one reading forward forever.

`merge_asof(..., direction='backward')` with no `tolerance` matches a bar to
the nearest earlier row NO MATTER HOW OLD. `ffill()` with no `limit` does the
same thing column-wise. Neither raises, neither logs, and both produce a full
column that looks like data.

This is not a hypothetical. It is the second of the three defect families this
project keeps meeting:

  Five news `*_available` flags were `notna()` over a forward-filled series, so
  they marked "this bar is inside the era we collected news for" rather than
  "there is a reading at this bar". They matched news_coverage to four decimal
  places on all three timeframes, which is how it was finally spotted.

  `cftc_available` was the constant 1.0 everywhere for the same reason.

  A resampler emitted zero-filled rows for EMPTY buckets, so a match always
  existed and the absence of data became the number zero.

USAGE

    python scripts/diagnostics/unbounded_join_lint.py            # report
    python scripts/diagnostics/unbounded_join_lint.py --baseline # rewrite the ledger

There are 31 unbounded joins and 18 unbounded fills in `src/` today. A check
that failed on all of them would be a red line that can never go green, and
this project already learned what those do: they teach their reader to skim
the whole report. So this is a RATCHET. The baseline records how many
unbounded sites each file holds today; the test fails when a file gains one.
The count can go down and never up, and nobody has to fix 49 call sites this
afternoon to stop the fiftieth from being written tonight.
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'src'
BASELINE = Path(__file__).with_name('unbounded_join_baseline.json')

#: Directories whose contents are not live code.
SKIP_PARTS = {'archive', 'dead_pipeline_code', '__pycache__', '.archive_docs'}

#: call name -> the keyword argument that bounds it
BOUNDED_BY = {
    'merge_asof': 'tolerance',
    'ffill': 'limit',
    'pad': 'limit',
    'bfill': 'limit',
    'backfill': 'limit',
}


def _call_name(node: ast.Call) -> str | None:
    func = node.func
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return None


def _is_ffill_via_fillna(node: ast.Call) -> bool:
    """`fillna(method='ffill')` is the same act spelled differently."""
    if _call_name(node) != 'fillna':
        return False
    for keyword in node.keywords:
        if keyword.arg == 'method' and isinstance(keyword.value, ast.Constant):
            return str(keyword.value.value) in {'ffill', 'pad', 'bfill', 'backfill'}
    return False


def _has_keyword(node: ast.Call, name: str) -> bool:
    return any(k.arg == name for k in node.keywords)


def scan_file(path: Path) -> list[dict]:
    """Every unbounded join or fill in one file."""
    try:
        tree = ast.parse(path.read_text(encoding='utf-8'))
    except (SyntaxError, UnicodeDecodeError):
        return []

    found: list[dict] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _call_name(node)
        if _is_ffill_via_fillna(node):
            name, bound = 'fillna(method=ffill)', 'limit'
        elif name in BOUNDED_BY:
            bound = BOUNDED_BY[name]
        else:
            continue
        if _has_keyword(node, bound):
            continue
        found.append({'call': name, 'line': node.lineno, 'bound': bound})
    return found


def scan(source: Path = SOURCE) -> dict[str, list[dict]]:
    """{relative path: [unbounded sites]}, live code only."""
    results: dict[str, list[dict]] = {}
    for path in sorted(source.rglob('*.py')):
        if SKIP_PARTS & set(path.parts):
            continue
        hits = scan_file(path)
        if hits:
            results[path.relative_to(ROOT).as_posix()] = hits
    return results


def counts(source: Path = SOURCE) -> dict[str, int]:
    return {path: len(hits) for path, hits in scan(source).items()}


def load_baseline() -> dict[str, int]:
    if not BASELINE.exists():
        return {}
    return json.loads(BASELINE.read_text(encoding='utf-8'))


def regressions(current: dict[str, int], baseline: dict[str, int]) -> dict[str, tuple[int, int]]:
    """Files holding MORE unbounded sites than the ledger allows."""
    return {
        path: (baseline.get(path, 0), count)
        for path, count in current.items()
        if count > baseline.get(path, 0)
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', action='store_true',
                        help='rewrite the ledger from the current tree')
    args = parser.parse_args()

    found = scan()
    current = {path: len(hits) for path, hits in found.items()}
    total = sum(current.values())

    if args.baseline:
        BASELINE.write_text(json.dumps(dict(sorted(current.items())), indent=2) + '\n',
                            encoding='utf-8')
        print(f'baseline written: {len(current)} files, {total} unbounded sites')
        return 0

    baseline = load_baseline()
    allowed = sum(baseline.values())
    print(f'{total} unbounded joins/fills in {len(current)} files '
          f'(ledger allows {allowed})\n')

    for path, hits in found.items():
        limit = baseline.get(path, 0)
        mark = '  ' if len(hits) <= limit else '!!'
        calls = ', '.join(f'{h["call"]}:{h["line"]}' for h in hits[:6])
        extra = f' (+{len(hits) - 6} more)' if len(hits) > 6 else ''
        print(f'{mark} {len(hits):>3}  {path}')
        print(f'          {calls}{extra}')

    bad = regressions(current, baseline)
    if bad:
        print('\nNEW unbounded sites:')
        for path, (was, now) in sorted(bad.items()):
            print(f'  {path}: {was} -> {now}')
        return 1
    fixed = sum(baseline.get(p, 0) - current.get(p, 0)
                for p in baseline if baseline.get(p, 0) > current.get(p, 0))
    if fixed:
        print(f'\n{fixed} site(s) bounded since the ledger was written — '
              f'rerun with --baseline to lock the gain in.')
    print('\nno new unbounded joins')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
