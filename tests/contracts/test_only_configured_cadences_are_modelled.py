"""Collecting a cadence and modelling it are separate decisions, and now separable.

Until 2026-09-03 the pipeline could not express "collect this and do not model
it", and the owner's decision of that date needed exactly that.

WHY THE ASYMMETRY MATTERS ENOUGH TO BUILD A SWITCH FOR IT. Intraday bars
accumulate forward only -- Yahoo serves sixty days of 15m -- so a bar not
collected today is gone permanently, while a bar not modelled today can be
modelled tomorrow by editing one line. This project has already paid for that
lesson from both sides: a manual purge destroyed 44,315 intraday bars
(REGISTER #218) and the restore that followed put 42,755 scrambled ones back
(#228). Stopping collection is irreversible; stopping analysis is not.

WHY INTRADAY, on three measurements rather than a preference:

    R8        smallest annualised Sharpe those frames can distinguish from
              zero: 6.01 at 15m, 2.07 at 60m
    R26       100% of intraday bars sit inside the sealed period, against
              11.6% of the daily frame -- nothing there to explore with
    R22, R25  11bp of round-trip friction already destroys a DAILY book
              against a 1.7bp per-bar edge: gross Sharpe 2.24, net -4.35.
              Intraday is that turnover multiplied.

The third is the one that settles it, and it arrived last.

WHAT THE SWITCH MUST NOT DO is go quiet. A cadence that is collected,
enriched, written into the batch and never trained on is a third of a run
producing nothing, and `batch_metadata.json` will not mention it -- the batch
was delivered, the modelling simply skipped it. That silence is the shape of
#205, #229 and #240, so the skip is reported at WARNING with the count.
"""
from __future__ import annotations

import inspect
import re
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG = PROJECT_ROOT / "src" / "config" / "processing.yaml"


@pytest.fixture(scope="module")
def modeling() -> dict:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))

    def find(node):
        if isinstance(node, dict):
            if "pool_tickers" in node:
                return node
            for value in node.values():
                found = find(value)
                if found is not None:
                    return found
        return None

    block = find(config)
    assert block is not None, "the modeling config block is no longer findable"
    return block


def test_the_switch_exists_and_names_a_cadence(modeling):
    """Absent or empty would mean 'model everything', which is the behaviour
    this replaced -- fine as a default, but then the decision is not in force."""
    assert "timeframes" in modeling, (
        "modeling.timeframes is gone, so the pipeline can no longer express "
        "'collect this and do not model it' and the 2026-09-03 decision is "
        "not in effect"
    )
    assert modeling["timeframes"], (
        "modeling.timeframes is empty, which means every cadence is modelled"
    )


def test_intraday_is_not_modelled(modeling):
    """The decision itself. If this is relaxed, say why in REGISTER first:
    R8, R26 and R22/R25 each independently argue against intraday."""
    configured = {str(name) for name in modeling["timeframes"]}
    intraday = {"15m", "60m", "1h", "5m", "1m"} & configured
    assert not intraday, (
        f"intraday cadences {sorted(intraday)} are back in the modelling set. "
        f"Three measurements argue against it and the newest is the strongest: "
        f"friction destroys a daily book, and intraday is the same turnover "
        f"multiplied (CLAIMS R22, R25)."
    )
    assert "1d" in configured, "the daily frame must still be modelled"


def test_collection_is_untouched_by_the_decision():
    """The whole point of the asymmetry: analysis stops, collection does not.

    A bar not collected today cannot be collected later -- Yahoo serves sixty
    days of 15m -- so if this ever starts failing, intraday history is being
    thrown away rather than set aside.
    """
    collectors = yaml.safe_load(
        (PROJECT_ROOT / "src" / "config" / "collectors.yaml").read_text(encoding="utf-8")
    )

    def cadences(node) -> set[str]:
        found: set[str] = set()
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "timeframes" and isinstance(value, dict):
                    found |= {str(name) for name in value}
                found |= cadences(value)
        elif isinstance(node, list):
            for item in node:
                found |= cadences(item)
        return found

    collected = cadences(collectors)
    assert "15m" in collected, (
        "15m has been dropped from COLLECTION. Analysis was what the decision "
        "stopped; collection is irreversible and was to continue."
    )


def test_a_skipped_cadence_is_reported_not_swallowed():
    """Read from the source: reaching this branch needs a full enriched frame,
    and what must not regress is that the skip is audible."""
    from src.pipeline.stages.modeling.orchestrator import ModelingStage

    source = inspect.getsource(ModelingStage._iter_model_contexts)
    assert "skipped_cadences" in source, (
        "the modelling stage no longer counts the cadences it skips"
    )
    assert re.search(r"logger\.warning\([^)]*NOT modelled", source, re.S), (
        "a cadence collected, enriched, written to the batch and never "
        "trained on is now silent -- the shape of #205, #229 and #240"
    )
