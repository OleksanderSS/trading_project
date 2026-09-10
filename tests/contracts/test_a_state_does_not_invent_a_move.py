"""A `state_X` column may not report a move that `X` never made.

Found while checking a leak control after the rebuild of 2026-09-08 (R67).
`state_sentiment_ema_1d` went from constant to varying, which the control said
must not happen -- and the investigation ended somewhere better than a leak:
`sentiment_ema_1d` is identically 0.0 across all 623,398 explorable rows, and
TWO rows carry a state of -1 anyway. QQQ on 1999-03-10 and BILI on 2018-03-28,
which are those instruments' first-ever bars: a "moved down" verdict where no
prior value exists to move from.

Measured across every state/base pair on the daily frame: 30 state columns sit
on a base that never varies, and 11 of them claim a move -- 26 rows in total,
0.004%. Systematic in MECHANISM, negligible in EXTENT.

NOT chased further, and the reason is a cost rather than a shrug: pinning where
the non-zero enters needs the enricher's INPUT frame reconstructed, and the
batch only holds its output. Twenty-six rows change no measurement this project
makes. So it is pinned at its size instead, and the ratchet is the finding: if
it grows, something changed in the enricher and this says so on the next batch.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURES = PROJECT_ROOT / "data/colab/accumulated/main_database/features.parquet"

#: Measured 2026-09-10 on the batch built 2026-09-08. Eleven columns, ten of
#: them two rows each and `state_fear_greed_available_1d` six. Lower it when
#: the enricher stops doing this; never raise it.
CONTRADICTING_ROWS = 26

#: How many state columns even have a constant base to contradict. Reported so
#: a fall in the ceiling can be read as a fix rather than as the bases quietly
#: coming alive -- which would lower the count for the opposite reason.
CONSTANT_BASE_COLUMNS = 30

BLOCK = 60


@pytest.fixture(scope="module")
def contradictions():
    if not FEATURES.exists():
        pytest.skip("no batch on disk")
    from src.pipeline.sealed_period import SEAL_START

    frame = pd.read_parquet(FEATURES, columns=["datetime", "interval"])
    is_daily = frame["interval"].astype(str).eq("1d").to_numpy()
    stamps = pd.to_datetime(frame.loc[is_daily, "datetime"]).reset_index(drop=True)
    if stamps.dt.tz is not None:
        stamps = stamps.dt.tz_localize(None)
    mask = is_daily.copy()
    mask[mask] = (stamps < pd.Timestamp(SEAL_START).tz_localize(None)).to_numpy()
    del frame

    names = pq.ParquetFile(FEATURES).schema_arrow.names
    pairs = [(n, n[len("state_"):]) for n in names
             if n.startswith("state_") and n.endswith("_1d")
             and n[len("state_"):] in names]

    on_constant: dict[str, int] = {}
    for start in range(0, len(pairs), BLOCK):
        chunk = pairs[start:start + BLOCK]
        columns = sorted({c for pair in chunk for c in pair})
        block = pd.read_parquet(FEATURES, columns=columns).loc[mask]
        for state_name, base_name in chunk:
            base = block[base_name]
            present = base.notna()
            if not present.any() or base[present].nunique(dropna=True) > 1:
                continue
            state = pd.to_numeric(block[state_name], errors="coerce")
            on_constant[state_name] = int((state.fillna(0) != 0).sum())
        del block
    return on_constant


def test_a_state_on_a_never_moving_base_does_not_multiply(contradictions):
    total = sum(contradictions.values())
    guilty = {name: n for name, n in contradictions.items() if n}
    assert total <= CONTRADICTING_ROWS, (
        f"{total} rows carry a non-zero state while their base never varies in "
        f"the explorable period, ceiling is {CONTRADICTING_ROWS}. A state that "
        "reports a move a series never made is a signal manufactured from "
        "nothing.\n"
        + "\n".join(f"  {name}: {n}" for name, n in sorted(
            guilty.items(), key=lambda kv: -kv[1])))


def test_the_ceiling_is_not_falling_for_the_wrong_reason(contradictions):
    """A drop is a fix only if the constant bases are still constant.

    If the bases come alive -- a backfill lands, say -- they stop being
    constant, leave this scan, and the count falls without anything being
    fixed. Reporting how many columns are in scope makes the two
    distinguishable.
    """
    assert len(contradictions) <= CONSTANT_BASE_COLUMNS, (
        f"{len(contradictions)} state columns sit on a constant base, up from "
        f"{CONSTANT_BASE_COLUMNS}. More constant bases is R63's finding "
        "getting worse, whatever this file's own count does.")


def test_the_scan_found_pairs_at_all(contradictions):
    """A silent zero would pass both tests above and mean nothing."""
    assert len(contradictions) >= 20, (
        f"only {len(contradictions)} state/base pairs with a constant base "
        "were found. Either the batch changed shape or the pairing broke, and "
        "a ratchet that inspects nothing always passes.")
