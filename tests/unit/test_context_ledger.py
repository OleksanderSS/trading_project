"""A finished context may be replayed only when its data is unchanged.

The pooled run of 2026-08-31 died in its eighth hour and took every finished
context with it; the 15m frame was recomputed six times across runs 1-6 for
byte-identical numbers. The ledger makes a restart cheap.

It is also the exact shape of defect this project keeps removing from other
places -- "there is a file on disk, so the work must be done" -- so what these
tests pin is not that reuse works, but that it REFUSES: any change to the
data must force a retrain, and reuse must be something an operator turned on.
"""
import json

import numpy as np
import pandas as pd
import pytest

from src.pipeline.stages.modeling.context_ledger import SCHEMA, ContextLedger

KEY = "AAPL_1d_target_up_1d_normal"


def _frame(seed=0, rows=5000, extra_column=False):
    rng = np.random.default_rng(seed)
    index = pd.date_range("2023-01-01", periods=rows, freq="h", tz="UTC")
    frame = pd.DataFrame(
        rng.normal(size=(rows, 40)),
        columns=[f"f{i}" for i in range(40)],
        index=index,
    )
    frame["target_up_1d"] = (rng.normal(size=rows) > 0).astype(float)
    if extra_column:
        frame["f_new"] = rng.normal(size=rows)
    return frame


@pytest.fixture
def ledger(tmp_path):
    written = ContextLedger(tmp_path / "context_ledger.json")
    written.record(
        KEY, written.fingerprint(_frame(), "target_up_1d"),
        champion={"winner": "catboost", "metrics": {"score": 0.61}},
    )
    return written


def test_the_same_data_is_reused(ledger):
    entry = ledger.lookup(KEY, ledger.fingerprint(_frame(), "target_up_1d"))
    assert entry is not None
    assert entry["champion"]["winner"] == "catboost"


@pytest.mark.parametrize(
    "name,frame_factory",
    [
        ("a row was added", lambda: _frame(rows=4999)),
        ("a column was added", lambda: _frame(extra_column=True)),
        ("every value differs", lambda: _frame(seed=1)),
    ],
)
def test_changed_data_forces_a_retrain(ledger, name, frame_factory):
    fingerprint = ledger.fingerprint(frame_factory(), "target_up_1d")
    assert ledger.lookup(KEY, fingerprint) is None, name


def test_one_changed_cell_anywhere_forces_a_retrain(ledger):
    """Not only the sampled rows.

    The sample is strided -- every 245th row on a half-million-row frame --
    so on its own it would miss a single edit almost always. The column sums
    and missing-counts are what make any changed cell move the fingerprint.
    """
    for row in (0, 1, 2, 137, 4998):
        changed = _frame()
        changed.iloc[row, 3] += 1e-6
        assert ledger.lookup(
            KEY, ledger.fingerprint(changed, "target_up_1d")
        ) is None, f"an edit at row {row} slipped through"


def test_one_changed_target_value_forces_a_retrain(ledger):
    changed = _frame()
    changed.iloc[10, -1] = 1.0 - changed.iloc[10, -1]
    assert ledger.lookup(KEY, ledger.fingerprint(changed, "target_up_1d")) is None


def test_an_unknown_context_is_never_reused(ledger):
    fingerprint = ledger.fingerprint(_frame(), "target_up_1d")
    assert ledger.lookup("NVDA_15m_target_up_1d_normal", fingerprint) is None


def test_the_record_survives_a_restart(tmp_path, ledger):
    reopened = ContextLedger(tmp_path / "context_ledger.json")
    entry = reopened.lookup(KEY, reopened.fingerprint(_frame(), "target_up_1d"))
    assert entry is not None
    assert entry["outcome"] == "champion"


def test_a_refusal_is_remembered_as_a_refusal(tmp_path):
    written = ContextLedger(tmp_path / "context_ledger.json")
    fingerprint = written.fingerprint(_frame(), "target_up_1d")
    written.record(KEY, fingerprint,
                   refusal={"context": KEY, "reasons": "lost to the clock"})

    entry = ContextLedger(tmp_path / "context_ledger.json").lookup(KEY, fingerprint)
    assert entry["outcome"] == "no_champion"
    assert entry["champion"] is None
    assert entry["refusal"]["reasons"] == "lost to the clock"


def test_a_ledger_from_another_schema_is_ignored_not_trusted(tmp_path):
    path = tmp_path / "context_ledger.json"
    path.write_text(json.dumps({
        "schema": "something_else",
        "entries": {KEY: {"fingerprint": "x", "champion": {"winner": "ghost"}}},
    }), encoding="utf-8")

    assert ContextLedger(path).lookup(KEY, "x") is None


def test_an_unreadable_ledger_does_not_take_the_run_with_it(tmp_path):
    path = tmp_path / "context_ledger.json"
    path.write_text("{not json", encoding="utf-8")

    ledger = ContextLedger(path)
    assert ledger.lookup(KEY, "x") is None
    # And it must still be usable afterwards, or a corrupt file would disable
    # the ledger for every run that follows.
    ledger.record(KEY, "x", champion={"winner": "linear"})
    assert json.loads(path.read_text(encoding="utf-8"))["schema"] == SCHEMA
