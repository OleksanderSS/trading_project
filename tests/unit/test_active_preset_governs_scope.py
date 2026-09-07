"""assets.yaml declared an active preset and nothing read it.

`active_preset: default_volatile` has sat in assets.yaml with an explicit
ticker list all along, while _get_tickers_from_args_or_config took every
ticker from every sector. The two disagreed and the config lost.

It made no visible difference while the database held 24 tickers, because
Stage 3 can only enrich what was collected. Collection starting to work
ended that: the 2026-08-06 run put 112 tickers in the database, the next
prepare enriched 110, and the export went from 15,433 rows to 128,033.
Stage 4 would then have trained roughly five times the models -- the
previous continue run built 506 in two hours.

The collection stage already honoured active_preset, but only as a fallback
for when no ticker list was passed, and one always was. So the mechanism
existed at both ends and was reachable at neither.
"""
from __future__ import annotations

import logging

import pytest

from src.cli.pipeline_executor import PipelineExecutor


def _assets(active=None, presets=None, sectors=None):
    config = {"sectors": sectors or {"a": {"assets": ["AAA", "BBB", "CCC"]}}}
    if active is not None:
        config["active_preset"] = active
    if presets is not None:
        config["presets"] = presets
    return config


def test_the_named_preset_decides():
    config = _assets(
        active="small",
        presets={"small": {"tickers": ["AAA", "BBB"]}},
    )

    assert PipelineExecutor._active_preset_tickers(config) == ["AAA", "BBB"]


def test_no_preset_named_means_every_sector():
    """Empty is 'no opinion', and the caller falls back to all sectors."""
    assert PipelineExecutor._active_preset_tickers(_assets()) == []


def test_an_unresolvable_preset_name_is_reported_not_ignored(caplog):
    """Silently widening the run to every instrument in the file is the one
    thing this must not do quietly."""
    config = _assets(active="typo", presets={"small": {"tickers": ["AAA"]}})

    with caplog.at_level(logging.WARNING):
        result = PipelineExecutor._active_preset_tickers(config)

    assert result == []
    assert any("no such preset" in r.getMessage() for r in caplog.records)


def test_an_empty_preset_is_reported(caplog):
    config = _assets(active="empty", presets={"empty": {"tickers": []}})

    with caplog.at_level(logging.WARNING):
        result = PipelineExecutor._active_preset_tickers(config)

    assert result == []
    assert any("no tickers" in r.getMessage() for r in caplog.records)


def test_a_malformed_preset_entry_does_not_raise():
    config = _assets(active="broken", presets={"broken": "not-a-mapping"})

    assert PipelineExecutor._active_preset_tickers(config) == []


def test_the_shipped_preset_states_its_whole_scope():
    """22 tickers: the 18 in the preset plus the four XL* sector ETFs.

    Every export up to 2026-08-06 carried those 22 -- the ETFs arriving as
    the Yahoo collector's benchmark_tickers, with nothing narrowing the set
    afterwards. A preset listing only 18 would silently drop four instruments
    that had been modelled all along, whenever it is activated again.

    This used to call `_active_preset_tickers` and expect 22 back. It cannot:
    `active_preset` was set to null on 2026-08-24, deliberately and with a
    measured reason -- at 22 names nothing measures, the rank portfolio over
    the whole batch gave t = 0.85 and the best non-price feature t = 0.29, so
    the run models all 110. The resolver correctly returns [] and the test was
    asserting the state from before the decision. The preset itself is what
    this invariant is about, so the preset is what it reads.
    """
    from src.config.unified_config_manager import UnifiedConfigManager

    assets = UnifiedConfigManager().get_config("assets") or {}
    preset = (assets.get("presets") or {}).get("default_volatile") or {}
    tickers = set(preset.get("tickers") or [])

    assert len(tickers) == 22, f"the preset lists {len(tickers)} tickers"
    assert {"XLE", "XLF", "XLK", "XLV"} <= tickers, "the benchmark ETFs were dropped"


def test_a_null_active_preset_models_every_sector():
    """The state that null was chosen FOR, so a silent revert is visible.

    Setting `active_preset` back to a preset name is a legitimate decision,
    but it narrows the run from 110 names to 22 -- and 22 is the width at
    which three independent routes all measured nothing. This does not forbid
    the change; it makes it impossible to make by accident.
    """
    from src.config.unified_config_manager import UnifiedConfigManager

    assets = UnifiedConfigManager().get_config("assets") or {}
    scope = set(PipelineExecutor._active_preset_tickers(assets)) or set(
        PipelineExecutor._load_tickers_from_sectors(assets.get("sectors", {}))
    )

    assert assets.get("active_preset") is None, (
        f"active_preset is {assets.get('active_preset')!r}. That narrows the "
        "run to a preset; null models every sector, which is what 2026-08-24 "
        "chose after measuring that 22 names carry no signal (t = 0.85).")
    assert len(scope) == 110, (
        f"the sector universe is {len(scope)} names, not 110. Every breadth "
        "measurement in CLAIMS -- R33, R59, R60 -- is stated over 110.")


@pytest.mark.parametrize("explicit", [["ONE"], ["ONE", "TWO"]])
def test_explicit_tickers_still_win(explicit):
    """--tickers is the operator speaking directly and outranks the preset."""
    from src.config.unified_config_manager import UnifiedConfigManager

    class _Args:
        tickers = explicit
        test_ticker = None

    assert PipelineExecutor._get_tickers(_Args(), UnifiedConfigManager()) == explicit
