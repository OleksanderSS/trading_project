"""No NEW join or fill may carry one reading forward forever.

The second of this project's three defect families, in one sentence: an
unbounded `merge_asof` or `ffill` turns "we have not seen a reading since
March" into a full column of March's reading, and nothing raises. Five news
`*_available` flags and `cftc_available` were all this, and all of them read
as working features for months.

There are 22 such sites in live `src/` today. A test that failed on all of
them would be a red line that can never go green — and this repository already
paid for learning what those do, in `verify_batch`, which failed the economic
calendar every single run until it was taught to skip on a measurement. So
this is a RATCHET rather than a rule: the ledger records what each file holds
today, and the test fails only when a file gains a site.

The number can go down. It must never go up. Nobody has to bound 22 call sites
this afternoon to stop the twenty-third from being written tonight.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.diagnostics.unbounded_join_lint import (  # noqa: E402
    BASELINE, counts, load_baseline, regressions, scan_file,
)


@pytest.fixture(scope='module')
def current() -> dict[str, int]:
    return counts()


@pytest.fixture(scope='module')
def baseline() -> dict[str, int]:
    return load_baseline()


def test_the_ledger_exists(baseline):
    assert BASELINE.exists(), (
        f'{BASELINE} is missing. Regenerate it with '
        f'`python scripts/diagnostics/unbounded_join_lint.py --baseline`.'
    )
    assert baseline, 'the ledger is empty, so every check below passes vacuously'


def test_no_file_gained_an_unbounded_join(current, baseline):
    bad = regressions(current, baseline)
    assert not bad, (
        'These files gained an unbounded merge_asof/ffill:\n'
        + '\n'.join(f'  {path}: {was} -> {now}' for path, (was, now) in sorted(bad.items()))
        + '\n\nAn unbounded backward join matches a bar to the nearest earlier row no '
          'matter how old, and an unbounded ffill does the same column-wise. Pass an '
          'explicit `tolerance=` or `limit=`. If unbounded really is correct here, '
          'say why in a comment and rerun the lint with --baseline.'
    )


def test_the_ledger_is_not_stale(current, baseline):
    """A file that no longer holds any unbounded site must leave the ledger.

    Otherwise the ledger keeps granting permission that nothing needs, and the
    next reader cannot tell allowance from fact.
    """
    gone = sorted(path for path in baseline if path not in current)
    assert not gone, (
        f'{gone} are in the ledger but hold no unbounded sites any more. '
        f'Rerun with --baseline to lock the gain in.'
    )


def test_the_total_never_exceeds_the_ledger(current, baseline):
    assert sum(current.values()) <= sum(baseline.values())


class TestTheScannerItself:
    """A lint nobody has tested is a lint that reports what it feels like."""

    def _write(self, tmp_path: Path, body: str) -> Path:
        path = tmp_path / 'sample.py'
        path.write_text(body, encoding='utf-8')
        return path

    def test_an_unbounded_merge_asof_is_found(self, tmp_path):
        found = scan_file(self._write(tmp_path,
            'import pandas as pd\n'
            'out = pd.merge_asof(a, b, on="datetime", direction="backward")\n'))
        assert [h['call'] for h in found] == ['merge_asof']

    def test_a_bounded_merge_asof_is_not(self, tmp_path):
        found = scan_file(self._write(tmp_path,
            'import pandas as pd\n'
            'out = pd.merge_asof(a, b, on="datetime", tolerance=pd.Timedelta("1h"))\n'))
        assert found == []

    def test_an_unbounded_ffill_is_found(self, tmp_path):
        found = scan_file(self._write(tmp_path, 'out = df.ffill()\n'))
        assert [h['call'] for h in found] == ['ffill']

    def test_a_bounded_ffill_is_not(self, tmp_path):
        found = scan_file(self._write(tmp_path, 'out = df.ffill(limit=4)\n'))
        assert found == []

    def test_the_same_act_spelled_the_old_way_is_found(self, tmp_path):
        # fillna(method='ffill') is the identical operation, and a lint that
        # only knew one spelling would bless the other.
        found = scan_file(self._write(tmp_path, 'out = df.fillna(method="ffill")\n'))
        assert [h['call'] for h in found] == ['fillna(method=ffill)']

    def test_a_plain_fillna_is_not_a_forward_fill(self, tmp_path):
        found = scan_file(self._write(tmp_path, 'out = df.fillna(0)\n'))
        assert found == []

    def test_a_call_split_over_several_lines_is_still_seen(self, tmp_path):
        # This is why the scanner parses rather than greps: the real call sites
        # in this repository span four and five lines.
        found = scan_file(self._write(tmp_path,
            'out = pd.merge_asof(\n'
            '    left,\n'
            '    right,\n'
            '    on="datetime",\n'
            '    direction="backward",\n'
            ')\n'))
        assert [h['call'] for h in found] == ['merge_asof']

    def test_a_tolerance_split_over_several_lines_is_credited(self, tmp_path):
        found = scan_file(self._write(tmp_path,
            'out = pd.merge_asof(\n'
            '    left, right, on="datetime",\n'
            '    tolerance=pd.Timedelta(minutes=15),\n'
            ')\n'))
        assert found == []

    def test_a_file_that_does_not_parse_is_skipped_not_crashed(self, tmp_path):
        assert scan_file(self._write(tmp_path, 'def broken(:\n')) == []

    def test_bfill_counts_too(self, tmp_path):
        # A backward fill leaks the FUTURE into the past, which is worse.
        found = scan_file(self._write(tmp_path, 'out = df.bfill()\n'))
        assert [h['call'] for h in found] == ['bfill']
