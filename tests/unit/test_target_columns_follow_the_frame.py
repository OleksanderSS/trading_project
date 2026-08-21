"""A target's horizon resolves per frame. Its columns have to as well.

`target_hourly_breakout_1h` names one indicator, `BB_Upper_60m`, and a "1h"
horizon -- which is valid on 15-minute bars (four of them) and on hourly bars
(one), so the target legitimately runs on both. The feature stage suffixes
indicators with the frame's own `interval` column, so the band is
`BB_Upper_15m` on one frame and `BB_Upper_1h` on the other. `BB_Upper_60m`
exists on neither.

That target and `target_volatility_spike_1h` therefore failed on every frame
of every run and were absent from every batch. The log said so twice a run,
under ERROR, for long enough that the lines had become scenery.
"""

import pandas as pd
import pytest

from src.targets.timeframe_contract import resolve_column_for_frame


def _frame(*columns):
    return pd.DataFrame({name: [1.0, 2.0, 3.0] for name in columns})


def test_the_config_spelling_finds_the_fifteen_minute_column():
    frame = _frame("close", "BB_Upper_15m")
    assert resolve_column_for_frame("BB_Upper_60m", frame, "15m") == "BB_Upper_15m"


def test_the_config_spelling_finds_the_hourly_column():
    """The frame labels itself `1h`; the contract calls the same thing `60m`."""
    frame = _frame("close", "BB_Upper_1h")
    assert resolve_column_for_frame("BB_Upper_60m", frame, "60m") == "BB_Upper_1h"
    assert resolve_column_for_frame("BB_Upper_60m", frame, "1h") == "BB_Upper_1h"


def test_an_exact_match_always_wins():
    """A target reaching for another timeframe's indicator keeps it."""
    frame = _frame("BB_Upper_1d", "BB_Upper_15m")
    assert resolve_column_for_frame("BB_Upper_1d", frame, "15m") == "BB_Upper_1d"


def test_an_unsuffixed_column_is_found_too():
    frame = _frame("close", "ATR_14")
    assert resolve_column_for_frame("ATR_14_60m", frame, "15m") == "ATR_14"


def test_a_column_that_is_nowhere_is_reported_as_missing():
    """Not invented, not silently swapped for something else."""
    frame = _frame("close", "volume")
    assert resolve_column_for_frame("BB_Upper_60m", frame, "15m") is None


def test_a_plain_price_column_is_left_alone():
    frame = _frame("close", "BB_Upper_15m")
    assert resolve_column_for_frame("close", frame, "15m") == "close"


@pytest.mark.parametrize("timeframe", ["15m", "60m", "1h", "1d", None, "weird"])
def test_it_never_raises_on_an_unknown_timeframe(timeframe):
    frame = _frame("close")
    assert resolve_column_for_frame("ATR_14_60m", frame, timeframe) is None


# ------------------------------------------------------- through the orchestrator


def test_the_orchestrator_rewrites_the_params_it_passes_on():
    from src.targets.target_orchestrator import TargetOrchestrator

    frame = _frame("close", "BB_Upper_15m", "ATR_14_15m")
    params = {"base_col": "close", "indicator_col": "BB_Upper_60m", "shift": -4}

    resolved = TargetOrchestrator._resolve_column_params(
        params, frame, "15m", "target_hourly_breakout_1h"
    )
    assert resolved["indicator_col"] == "BB_Upper_15m"
    assert resolved["base_col"] == "close"
    assert resolved["shift"] == -4
    assert params["indicator_col"] == "BB_Upper_60m"   # caller's dict untouched


def test_an_unresolvable_column_is_left_for_the_calculator_to_reject():
    """Silence here would turn a missing column into a wrong number."""
    from src.targets.target_orchestrator import TargetOrchestrator

    frame = _frame("close")
    resolved = TargetOrchestrator._resolve_column_params(
        {"base_col": "ATR_14_60m", "shift": -1}, frame, "15m", "target_volatility_spike_1h"
    )
    assert resolved["base_col"] == "ATR_14_60m"


def test_the_two_targets_that_never_existed_now_build():
    """End to end on the shapes that failed: an ATR spike and a band break."""
    from src.targets.target_orchestrator import TargetOrchestrator

    bars = pd.date_range("2026-03-02 09:30", periods=40, freq="15min")
    frame = pd.DataFrame({
        "ticker": ["AAPL"] * 40,
        "datetime": bars,
        "interval": ["15m"] * 40,
        "close": [100.0 + i for i in range(40)],
        "BB_Upper_15m": [110.0] * 40,
        "ATR_14_15m": [1.0 + 0.1 * i for i in range(40)],
    })

    targets = {
        "target_hourly_breakout_1h": {
            "type": "classification_binary",
            "params": {"horizon": "1h", "shift": -4, "base_col": "close",
                       "indicator_col": "BB_Upper_60m", "threshold": 0.0},
        },
        "target_volatility_spike_1h": {
            "type": "classification_binary",
            "params": {"shift": -4, "base_col": "ATR_14_60m", "threshold": 0.03},
        },
    }
    out = TargetOrchestrator(targets, timeframe="15m").generate_targets(frame, timeframe="15m")

    for name in targets:
        assert name in out.columns, f"{name} still missing"
        assert out[name].notna().any(), f"{name} is entirely NaN"


# ------------------------------------------------------------- the real config


def _configured(name):
    import io as _io

    import yaml

    cfg = yaml.safe_load(_io.open('src/config/targets.yaml', encoding='utf-8'))
    return (cfg.get('targets') or cfg)[name]['params']


def test_the_two_broken_targets_no_longer_name_a_column_that_exists_nowhere():
    """`_60m` was never produced by any frame: the hourly one labels itself `1h`."""
    assert '60m' not in str(_configured('target_volatility_spike_1h')['base_col'])
    assert '60m' not in str(_configured('target_hourly_breakout_1h')['indicator_col'])


def test_the_hourly_volatility_target_is_not_a_copy_of_the_fifteen_minute_one():
    """Resolving the suffix collapsed them onto each other until a horizon was set."""
    hourly = _configured('target_volatility_spike_1h')
    intraday = _configured('target_volatility_spike_15m')

    assert hourly.get('horizon') == '1h'
    assert hourly['shift'] != intraday['shift']


def test_both_targets_build_on_the_frame_the_pipeline_runs_them_on():
    """Stage 3 generates these on the 15-minute frame, where a 1h horizon is 4 bars."""
    from src.targets.target_orchestrator import TargetOrchestrator

    bars = pd.date_range('2026-03-02 09:30', periods=60, freq='15min')
    frame = pd.DataFrame({
        'ticker': ['AAPL'] * 60,
        'datetime': bars,
        'interval': ['15m'] * 60,
        'close': [100.0 + (i % 7) for i in range(60)],
        'BB_Upper_15m': [104.0] * 60,
        'ATR_14_15m': [1.0 + 0.01 * (i % 5) for i in range(60)],
    })
    configs = {
        name: {'type': 'classification_binary', 'params': _configured(name)}
        for name in ('target_volatility_spike_1h', 'target_hourly_breakout_1h')
    }
    out = TargetOrchestrator(configs, timeframe='15m').generate_targets(frame, timeframe='15m')

    for name in configs:
        assert name in out.columns, f'{name} still missing'
        assert out[name].notna().any(), f'{name} is entirely NaN'
        assert out[name].nunique() > 1, f'{name} is constant, so it teaches nothing'
