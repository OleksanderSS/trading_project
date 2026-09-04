"""The invariant script must be able to gate a run, or it will not be run.

REGISTER #170 asked why `batch_invariants.py` gates nothing when its own
docstring says "Exit code is non-zero when any check fails, so it can gate a
run". Measured 2026-09-04: **zero callers** anywhere in `src/`, `tests/` or CI.

Running it explained why. It exited 1 on two checks whose findings are known,
explained, and not corruption:

    13 columns 98% on a truthful ZERO -- the news and sentiment family, which
       has no data before 2024 and says so through its own `*_available` flag
       (CLAIMS R29). The median check's comment already noted that a sparse
       count column and a fabricated fill both trip it and only the second is
       a defect; it just never acted on the distinction.

    69 features constant during training -- the same sources, seen from the
       other side.

A check that fails on a condition everyone has agreed to is a check nobody
runs. And this one held, unrun, the answers to a full day of measurement: both
failures are exactly what R29 and gate A established from scratch hours later.

WHAT CHANGED: a failure is now either CORRUPTION -- the data is wrong and a
rebuild must not carry it -- or ADVISORY -- the data is right and thin. Only
corruption sets the exit code. On the current batch that is 0 blocking, 1
advisory, exit 0.

WHAT IS STILL THE OWNER'S: whether the pipeline itself calls this and stops on
a blocking failure. That decides when runs die, which is the same class as
#229 and #252. Making the verdict meaningful is what makes the decision
available; taking it would be deciding for them.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT_ROOT / "scripts" / "diagnostics" / "batch_invariants.py"


def _module():
    # Registered in sys.modules BEFORE exec: `@dataclass` resolves its
    # annotations through `sys.modules[cls.__module__]`, and a module loaded
    # from a path without registering is not there yet.
    import sys

    spec = importlib.util.spec_from_file_location("batch_invariants", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["batch_invariants"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def invariants():
    assert SCRIPT.exists(), f"{SCRIPT.relative_to(PROJECT_ROOT)} is gone"
    return _module()


def test_a_result_can_say_it_is_not_corruption(invariants):
    """Without this field every failure is the same failure, and the script
    exits 1 on a batch everyone agrees is fine."""
    result = invariants.Result("x", False, "detail", "story")
    assert result.blocking is True, "the default must be strict"
    assert invariants.Result("x", False, "d", "s", blocking=False).blocking is False


def test_the_learnability_check_is_advisory(invariants):
    """A feature constant in the training window is thin data, not wrong data.
    Blocking on it is why nobody ran this script."""
    import inspect

    # The function's own source, not a window from the first mention of the
    # name: the check has an early return carrying the same string, and a
    # fixed-size window from there never reaches the verdict.
    source = inspect.getsource(invariants.check_features_learnable)
    assert "blocking=False" in source, (
        "the learnability check blocks again, so the script exits 1 on every "
        "batch whose newer sources have no pre-seal history"
    )


def test_the_exit_code_counts_only_corruption(invariants):
    source = SCRIPT.read_text(encoding="utf-8")
    assert "return 1 if blocking else 0" in source, (
        "the exit code is back to counting advisories, which is what made the "
        "script unwireable"
    )
    assert "advisory" in source, (
        "advisories are no longer reported; silence is not the fix for a "
        "check that cried wolf"
    )


def _parquet(tmp_path: Path) -> tuple[Path, pd.DataFrame]:
    rows = 20_000
    rng = np.random.default_rng(3)
    spread = rng.normal(size=rows).round(4)

    fabricated = spread.copy()
    fabricated[: int(rows * 0.4)] = 313.569          # a whole-frame median fill
    fabricated[int(rows * 0.4):] += 313.569

    sparse = spread.copy()
    sparse[: int(rows * 0.98)] = 0.0                 # a truthful zero

    frame = pd.DataFrame({
        "datetime": pd.date_range("2000-01-01", periods=rows, freq="h", tz="UTC"),
        "ticker": "AAA",
        "interval": "1d",
        "fabricated_col": fabricated,
        "sparse_zero_col": sparse,
    })
    path = tmp_path / "batch.parquet"
    frame.to_parquet(path)
    return path, frame


def test_a_pile_on_a_truthful_zero_is_not_counted_as_fabrication(tmp_path, invariants):
    """The distinction the check's own comment described and did not act on."""
    path, frame = _parquet(tmp_path)
    result = invariants.check_nothing_is_mostly_its_median(path, frame)

    assert "sparse_zero_col" not in " ".join(
        result.detail.split(";")[:1]
    ), "a column piled on zero is being reported as a fabricated median"
    assert "truthful 0" in result.detail, (
        "the zero pile-ups vanished from the output entirely; that trades one "
        "silence for another"
    )
    assert "sparse_zero_col" in result.detail, (
        "the zero pile-up is not named anywhere, so the reader cannot check it"
    )


def test_a_fabricated_median_still_fails(tmp_path, invariants):
    """The mirror: the check must still catch what it was written for --
    70.5% of FRED_INDPRO_1d was exactly the column median, a constant computed
    over the whole frame, so it contained the future."""
    path, frame = _parquet(tmp_path)
    result = invariants.check_nothing_is_mostly_its_median(path, frame)

    assert result.ok is False, "a whole-frame median fill no longer fails"
    assert "fabricated_col" in result.detail
    assert result.blocking is True, "fabricated data must block a rebuild"


def test_the_session_check_names_the_ticker_not_just_a_count(invariants, tmp_path):
    """REGISTER #164 read as "SMA_20 disagrees on 3.4% of hourly rows" for six
    days. It was ONE ticker carrying every out-of-session bar in the frame, and
    a count alone would have left that to be rediscovered."""
    rows = 3_000
    session = [14, 15, 16, 17, 18, 19, 20]
    stamps, tickers = [], []
    base = pd.Timestamp("2026-01-01", tz="UTC")
    for day in range(rows // len(session)):
        for hour in session:
            for name in ("AAA", "BBB"):
                stamps.append(base + pd.Timedelta(days=day, hours=hour))
                tickers.append(name)
    frame = pd.DataFrame({"ticker": tickers, "_time": stamps})
    # One ticker also trades pre-market, exactly as AAPL does.
    extra = pd.DataFrame({
        "ticker": "AAA",
        "_time": [base + pd.Timedelta(days=d, hours=9) for d in range(40)],
    })
    frame = pd.concat([frame, extra], ignore_index=True)

    result = invariants.check_all_tickers_share_a_session(tmp_path, frame)
    assert result.ok is False
    assert "AAA" in result.detail, "the culprit is not named"
    assert "BBB" not in result.detail, "a clean ticker is being blamed"
    assert result.blocking is True


def test_a_daily_frame_has_no_session_to_disagree_about(invariants, tmp_path):
    """Every daily bar sits at the same hour. A check that fired here would
    fail every rebuild on the one frame that is actually gated."""
    frame = pd.DataFrame({
        "ticker": ["AAA", "BBB"] * 500,
        "_time": list(pd.date_range("2020-01-01", periods=1000, tz="UTC")) ,
    })
    result = invariants.check_all_tickers_share_a_session(tmp_path, frame)
    assert result.ok is True
    assert "daily" in result.detail


def test_the_script_is_called_by_the_pipeline(invariants):
    """It was called by nothing for its whole life (#170). Now the feature
    stage runs it after each checkpoint, and that must not quietly come undone
    -- a gate nobody invokes is the state this file was written to end."""
    callers = []
    for root in ("src", "scripts/ci"):
        base = PROJECT_ROOT / root
        if not base.exists():
            continue
        for path in base.rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            if "batch_invariants" in path.read_text(encoding="utf-8", errors="replace"):
                callers.append(str(path.relative_to(PROJECT_ROOT)))

    assert callers, (
        "batch_invariants.py has no caller in src/ again. It spent its whole "
        "life in that state while holding the answers to a day of measurement "
        "(REGISTER #170)."
    )
