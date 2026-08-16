"""Run the real enricher chain on real data and report what each step produces.

Written because testing enrichers in isolation lied twice in one session, in
two different ways:

  news_impact   passed a unit test whose fixture put the article body in
                `text`. The live frame keeps it in `title` and leaves `text`
                empty on all 15,836 rows, so the enricher scored blanks and
                shipped [0.000, 0.000] to every model.

  sentiment_features  appeared to "add nothing" when called alone. In the
                chain, nlp_features runs first and supplies the very column it
                looks for, and it adds 11 features without complaint.

Both failures share a cause: an enricher is not a pure function of its own
code. It is a function of the code, the SHAPE of the real data, and everything
that ran before it. So this harness changes only two things versus a unit
test, and those two things are the whole point — it uses the batch's own
artifacts instead of fixtures, and it runs the enrichers in their real
priority order, feeding each the previous one's output.

What it reports per step, and why each column earns its place:

  +cols       how many columns appeared. Zero means the enricher did nothing.
  const       of those, how many never vary. A constant column cannot inform
              a model, and this is how hype, news_impact, market_phase and the
              context features all failed — loudly successful, silently empty.
              READ THIS ONE WITH THE SAMPLE SIZE IN MIND. `--rows` caps the
              bars, and a feature computed over an EXPANDING window needs
              history before it can call anything unusual. On the full batch
              `news_significance_level` takes 2 values on 15m and 3 on 60m and
              1d; on a 4,000-row sample it is flat, because nothing has yet
              exceeded the 90th percentile of what came before it. A constant
              here is a question, not a verdict — check the column against the
              real batch before treating it as a defect. That cost two
              investigations to learn.
  empty       all-NaN columns: attached to nothing at all.
  rows        a change means the enricher filtered; legitimate, but it breaks
              any positional reattachment downstream, so it must be visible.
  order       whether row order survived. A reorder plus a dropped identity
              column is what put 54,000 bars on the wrong dates in August.

Usage:

    python scripts/diagnostics/enricher_chain_report.py            # 15m
    python scripts/diagnostics/enricher_chain_report.py --tf 1d
    python scripts/diagnostics/enricher_chain_report.py --rows 5000

Reads only from data/processed/*.parquet, so it neither needs the database nor
disturbs a running pipeline.
"""
from __future__ import annotations

import argparse
import glob
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def _latest(pattern: str) -> Path | None:
    hits = sorted(glob.glob(pattern))
    return Path(hits[-1]) if hits else None


def _load(timeframe: str, rows: int) -> tuple[pd.DataFrame, dict]:
    bars_path = _latest(f"data/processed/prices_{timeframe}_*.parquet")
    if bars_path is None:
        raise SystemExit(
            f"No data/processed/prices_{timeframe}_*.parquet. "
            f"Run stage 0-3 at least once first."
        )
    bars = pd.read_parquet(bars_path)
    if rows and len(bars) > rows:
        # Keep whole tickers rather than a head slice: several enrichers group
        # by ticker, and a head slice silently hands them one.
        keep = bars["ticker"].drop_duplicates().tolist()
        per = max(1, rows // max(1, len(keep)))
        bars = (bars.sort_values("datetime")
                    .groupby("ticker", group_keys=False)
                    .apply(lambda g: g.tail(per)))
    kwargs: dict = {}
    for key, pattern in (("news", "data/processed/news_*.parquet"),
                         ("macro_data", "data/processed/macro_data_*.parquet")):
        p = _latest(pattern)
        if p is not None:
            kwargs[key] = pd.read_parquet(p)
    kwargs.update(_from_database())
    return bars.reset_index(drop=True), kwargs


def _from_database() -> dict:
    """Market-wide tables that stage 2 does not write to data/processed.

    Read-only and best-effort: a pipeline holding the database must not stop
    this report, and its absence only means those enrichers have nothing to
    show. Keyed under both spellings for the same reason stage 3 is — the
    tables are named `cftc_data` while enrichers ask for `cftc`.
    """
    wanted = ("cftc_data", "fear_greed_data", "wikipedia_attention_data",
              "insider_trades", "sociological_sentiment_data",
              "economic_calendar")
    out: dict = {}
    try:
        import duckdb

        con = duckdb.connect("data/trading_data.duckdb", read_only=True)
    except Exception as exc:  # noqa: BLE001 - the DB is optional here
        print(f"  (database not read: {type(exc).__name__} — "
              f"market-wide enrichers will show nothing)")
        return out
    try:
        present = {r[0] for r in con.execute("show tables").fetchall()}
        for table in wanted:
            if table not in present:
                continue
            frame = con.execute(f"select * from {table}").df()
            if frame.empty:
                continue
            out[table] = frame
            if table.endswith("_data"):
                out.setdefault(table[:-5], frame)
    finally:
        con.close()
    return out


def _score_news(kwargs: dict) -> dict:
    """Stage 3 scores the news before any enricher sees it.

    Skipping this is not a smaller test, it is a different one: the sentiment
    column is empty in the stored artifact and is filled here, so an enricher
    that reads it would be measured against blanks.
    """
    news = kwargs.get("news")
    if news is None:
        return kwargs
    try:
        from src.pipeline.stages.feature_engineering.orchestrator import (
            FeatureEngineeringStage,
        )
        kwargs = dict(kwargs)
        kwargs["news"] = FeatureEngineeringStage._score_news_sentiment(news.copy())
    except Exception as exc:  # noqa: BLE001 - diagnostic must not die here
        print(f"  (news sentiment scoring skipped: {type(exc).__name__}: {exc})")
    return kwargs


class _Capture(logging.Handler):
    """Collect WARNING+ records emitted while one enricher runs.

    BaseEnricher.enrich() is a template method that catches its own
    exceptions and returns the input frame unchanged. So a failure never
    reaches this script as an exception — it arrives as a frame that did not
    grow, and the only account of what went wrong is a log line. Without this
    the report can say "ADDED NOTHING" and nothing else, which is the same
    silence the harness exists to break.
    """

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.records: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.records.append(f"{record.levelname} {record.getMessage()}")
        except Exception:  # noqa: BLE001 - a broken log line must not stop the run
            pass


def _bar_dates(frame: pd.DataFrame) -> pd.Series | None:
    """Each row's own timestamp, keyed by the collector hash."""
    if "hash" not in frame.columns or "datetime" not in frame.columns:
        return None
    stamps = pd.to_datetime(frame["datetime"], errors="coerce", utc=True)
    series = pd.Series(stamps.dt.tz_localize(None).to_numpy(),
                       index=frame["hash"].to_numpy())
    return series[~series.index.duplicated()]


def _dates_intact(truth: pd.Series | None, after: pd.DataFrame) -> float | None:
    """Share of rows whose datetime is still the one their own bar carried.

    The order check beside this one asks whether the rows moved. This asks the
    question that actually matters, and they are not the same question: an
    enricher can put every row back in place and still have overwritten the
    timestamps, which is precisely what happened to 15m on the v14 rebuild --
    24,143 of 26,295 bars carrying somebody else's time while row count, row
    order and every hash were perfect. Anchored on the collector hash, so it
    survives reordering, reindexing and index-to-column rescues alike.
    """
    if truth is None:
        return None
    current = _bar_dates(after)
    if current is None or current.empty:
        return None
    shared = current.index.intersection(truth.index)
    if not len(shared):
        return None
    return float((current[shared] == truth[shared]).mean())


def _profile(before: pd.DataFrame, after: pd.DataFrame) -> dict:
    added = [c for c in after.columns if c not in before.columns]
    const, empty = [], []
    for col in added:
        series = pd.to_numeric(after[col], errors="coerce")
        if series.isna().all():
            empty.append(col)
        elif series.nunique(dropna=True) <= 1:
            const.append(col)
    order_kept = True
    if len(before) == len(after):
        try:
            order_kept = bool((before.index == after.index).all())
        except (ValueError, TypeError):
            order_kept = False
    return {
        "added": added, "const": const, "empty": empty,
        "rows_before": len(before), "rows_after": len(after),
        "order_kept": order_kept,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tf", default="15m", help="timeframe: 15m, 60m, 1d")
    ap.add_argument("--rows", type=int, default=6000,
                    help="cap on bars, spread across all tickers (0 = all)")
    args = ap.parse_args()

    # Quiet the CONSOLE, not the loggers. Raising a logger's level would stop
    # records reaching _Capture too, and those records are the whole reason
    # this script can say why an enricher added nothing rather than only that
    # it did. So the existing handlers are muted and ours is added instead.
    from src.config.unified_config_manager import get_current_config
    from src.features.feature_orchestrator import FeatureOrchestrator

    # Muted AFTER the imports: ProjectLogger installs its handlers on import,
    # so anything done before this line is undone by them.
    _root = logging.getLogger()
    for _handler in list(_root.handlers):
        _handler.setLevel(logging.CRITICAL)
    _root.setLevel(logging.INFO)

    frame, kwargs = _load(args.tf, args.rows)
    kwargs = _score_news(kwargs)

    config = get_current_config()
    orchestrator = FeatureOrchestrator.create_from_config(config)
    enrichers = sorted(orchestrator.enrichers, key=lambda e: e.priority)

    print(f"\nChain report — {args.tf}: {len(frame)} bars, "
          f"{frame['ticker'].nunique()} tickers, {len(enrichers)} enrichers")
    for key, value in kwargs.items():
        print(f"  input {key}: {len(value)} rows")
    print()
    truth = _bar_dates(frame)
    if truth is None:
        print("  (no hash/datetime pair — date integrity cannot be checked)")
    head = (f"{'#':>2} {'enricher':22s} {'prio':>4s} {'+cols':>6s} "
            f"{'const':>6s} {'empty':>6s} {'rows':>12s} {'order':>6s} "
            f"{'dates':>6s}  note")
    print(head)
    print("-" * len(head))

    problems: list[str] = []
    complaints: dict[str, list[str]] = {}
    for i, enricher in enumerate(enrichers, 1):
        before = frame.copy()
        _cap = _Capture()
        _root.addHandler(_cap)
        note = ""
        try:
            # The orchestrator's own per-enricher path, not enricher.enrich()
            # directly: it repairs the row index between steps, and two
            # enrichers only survive because of that repair. Calling them raw
            # reported "ADDED NOTHING" for context_map and market_context,
            # which is true of the enricher alone and false of the pipeline —
            # exactly the kind of half-truth this harness exists to avoid.
            frame, _stats = orchestrator._process_single_enricher(
                enricher, frame, kwargs
            )
        except Exception as exc:  # noqa: BLE001 - the report is the product
            note = f"RAISED {type(exc).__name__}: {str(exc)[:60]}"
            problems.append(f"{enricher.name}: {note}")
            frame = before
        finally:
            _root.removeHandler(_cap)
        if _cap.records:
            complaints[enricher.name] = _cap.records
        p = _profile(before, frame)
        rows = (f"{p['rows_before']}" if p["rows_before"] == p["rows_after"]
                else f"{p['rows_before']}->{p['rows_after']}")
        if not note:
            if not p["added"]:
                note = "ADDED NOTHING"
                problems.append(f"{enricher.name}: added no columns")
            elif len(p["const"]) + len(p["empty"]) == len(p["added"]):
                note = "ALL CONSTANT/EMPTY"
                problems.append(f"{enricher.name}: every column constant or empty")
            elif p["const"] or p["empty"]:
                note = ", ".join((p["const"] + p["empty"])[:3])
        intact = _dates_intact(truth, frame)
        if intact is None:
            dates = "-"
        elif intact >= 1.0:
            dates = "ok"
        else:
            dates = f"{intact:.1%}"
            problems.append(
                f"{enricher.name}: {1 - intact:.1%} of bars now carry a "
                f"datetime that is not their own"
            )
        print(f"{i:2d} {enricher.name[:22]:22s} {enricher.priority:4d} "
              f"{len(p['added']):6d} {len(p['const']):6d} {len(p['empty']):6d} "
              f"{rows:>12s} {'ok' if p['order_kept'] else 'CHANGED':>6s} "
              f"{dates:>6s}  {note}")

    print(f"\nfinal frame: {frame.shape[0]} rows x {frame.shape[1]} columns")
    if problems:
        print(f"\n{len(problems)} enricher(s) need attention:")
        for line in problems:
            print(f"  - {line}")
    else:
        print("\nEvery enricher added at least one varying column.")
    if complaints:
        print("\nWhat they complained about while running:")
        for _name, _lines in complaints.items():
            print(f"\n  {_name}")
            for _line in dict.fromkeys(_lines):
                print(f"    {_line[:190]}")

    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
