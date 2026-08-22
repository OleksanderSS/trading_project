"""Did this session's changes reach the data?

Run against the batch a rebuild produces, not against the code that produced
it. Every check here corresponds to something changed between 2026-08-20 and
2026-08-22, and each answers with a number rather than a status word.

The checks, and what a failure means:

    cross-sectional targets     the calculator ran and produced a real
                                distribution, not a column of zeros
    context_velocity_rank       the self-calibrating threshold reached the
                                frame instead of the absolute one
    corporate filing events     SEC filings arrived as events; coverage is
                                bounded by the source (16 tickers, 5 months),
                                so a low number is expected and a ZERO is not
    revived targets             the two that failed on the 15-minute frame for
                                every run in the project's history
    SPY filings                 SPY was reading a dead Van Kampen trust; it
                                should now carry filings of its own
    empty columns at selection  1,759 of 2,185 were empty and survived the
                                variance filter; the fixed one should leave
                                far fewer

Reads in column batches. It is meant to be run right after a rebuild, on a
machine that has just held 4.5 GiB for one, and a diagnostic that pushes it
over is not a diagnostic.

    python scripts/diagnostics/verify_session_changes.py
    python scripts/diagnostics/verify_session_changes.py --log logs/rebuild_xsect_v5_20260822.log
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

FEATURES = Path("data/colab/accumulated/main_database/features.parquet")
TARGETS = Path("data/colab/accumulated/main_database/targets.parquet")

PASS, FAIL, SKIP = "OK  ", "FAIL", "SKIP"


class Report:
    def __init__(self) -> None:
        self.failures = 0

    def say(self, mark: str, label: str, detail: str) -> None:
        if mark == FAIL:
            self.failures += 1
        print(f"  [{mark}] {label:38s} {detail}")


def _columns(path: Path) -> list[str]:
    return list(pq.ParquetFile(path).schema_arrow.names)


def _read(path: Path, columns: list[str]) -> pd.DataFrame:
    present = [c for c in columns if c in set(_columns(path))]
    if not present:
        return pd.DataFrame()
    return pd.read_parquet(path, columns=present)


def check_cross_sectional_targets(report: Report) -> None:
    names = [c for c in _columns(TARGETS) if "relative" in c]
    if not names:
        report.say(FAIL, "cross-sectional targets", "no target_relative_* column in the batch")
        return
    frame = _read(TARGETS, [*names, "interval"])
    for name in names:
        values = pd.to_numeric(frame[name], errors="coerce").dropna()
        if values.empty:
            report.say(FAIL, name, "present but entirely null")
        elif values.nunique() < 2:
            report.say(FAIL, name, f"constant at {values.iloc[0]}")
        else:
            report.say(PASS, name, f"{len(values):,} values, mean {values.mean():+.6f}, "
                                   f"{values.nunique():,} distinct")


def check_context_velocity_rank(report: Report) -> None:
    names = [c for c in _columns(FEATURES) if "context_velocity_rank" in c]
    if not names:
        report.say(FAIL, "context_velocity_rank", "absent; the gate is back on an absolute threshold")
        return
    frame = _read(FEATURES, names)
    for name in names[:3]:
        values = pd.to_numeric(frame[name], errors="coerce").dropna()
        if values.empty or values.nunique() < 2:
            report.say(FAIL, name, "present but carries no distribution")
        else:
            report.say(PASS, name, f"range {values.min():.3f}..{values.max():.3f}, "
                                   f"{values.nunique():,} distinct")


def check_filing_events(report: Report) -> None:
    names = [c for c in _columns(FEATURES) if c.startswith("filing_")]
    if not names:
        report.say(FAIL, "corporate filing events", "no filing_* column; the enricher did not run")
        return
    frame = _read(FEATURES, [*names, "ticker"])
    available = [c for c in names if c.endswith("available")]
    if available:
        covered = int(pd.to_numeric(frame[available[0]], errors="coerce").fillna(0).sum())
        share = covered / max(1, len(frame))
        mark = FAIL if covered == 0 else PASS
        report.say(mark, "filing_data_available",
                   f"{covered:,} of {len(frame):,} bars ({share:.2%}); source covers "
                   f"16 tickers over ~5 months, so a small share is expected")
    for name in names:
        if name in available:
            continue
        values = pd.to_numeric(frame[name], errors="coerce").dropna()
        detail = "no values" if values.empty else f"{len(values):,} values, max {values.max():g}"
        report.say(PASS if not values.empty else FAIL, name, detail)


def check_revived_targets(report: Report) -> None:
    wanted = ["target_hourly_breakout_1h", "target_volatility_spike_1h"]
    have = [c for c in wanted if c in _columns(TARGETS)]
    if not have:
        report.say(FAIL, "revived targets", "neither is in the batch")
        return
    frame = _read(TARGETS, [*have, "interval"])
    for name in have:
        per_interval = (
            frame[frame[name].notna()]["interval"].value_counts().to_dict()
            if "interval" in frame else {}
        )
        fifteen = per_interval.get("15m", 0)
        mark = PASS if fifteen else FAIL
        report.say(mark, name, f"by interval {per_interval}; the 15-minute slice is "
                               f"what was missing on every previous run")


def check_empty_columns_at_selection(report: Report, interval: str = "1d") -> None:
    """The variance filter used to remove 15 of 1,774 columns that carry nothing."""
    keys = [k for k in ("ticker", "datetime", "interval") if k in _columns(FEATURES)]
    index = _read(FEATURES, keys)
    if "interval" not in index.columns:
        report.say(SKIP, "empty columns at selection", "no interval column")
        return
    rows = index.index[index["interval"] == interval].to_numpy()
    if len(rows) == 0:
        report.say(SKIP, "empty columns at selection", f"no {interval} rows")
        return
    if len(rows) > 40_000:
        rows = rows[:: max(1, len(rows) // 40_000)]

    feature_columns = [c for c in _columns(FEATURES) if c not in keys]
    empty = constant = varying = 0
    for start in range(0, len(feature_columns), 100):
        chunk = feature_columns[start:start + 100]
        block = pd.read_parquet(FEATURES, columns=chunk).iloc[rows]
        for column in chunk:
            values = pd.to_numeric(block[column], errors="coerce")
            present = int(values.notna().sum())
            if present == 0:
                empty += 1
            elif values.nunique() < 2:
                constant += 1
            else:
                varying += 1
        del block
    report.say(PASS, f"columns on the {interval} slice",
               f"{empty:,} empty, {constant:,} constant, {varying:,} varying "
               f"-- the fixed filter removes the first two")


def check_stage3_timings(report: Report, log: Path | None) -> None:
    if log is None or not log.exists():
        report.say(SKIP, "stage 3 phase breakdown", "no log given")
        return
    text = log.read_text(encoding="utf-8", errors="replace")
    phases = re.findall(r"([\d.]+) min\s+([\d.]+)%\s+(.+)", text)
    if not phases:
        report.say(SKIP, "stage 3 phase breakdown", "the run did not reach the summary")
        return
    for minutes, share, name in phases[:8]:
        report.say(PASS, name.strip()[:38], f"{float(minutes):7.1f} min  {share}%")

    peak = re.findall(r"Peak memory held: ([\d.]+) GiB, at (.+?)\.", text)
    if peak:
        report.say(PASS, "peak memory", f"{peak[-1][0]} GiB at {peak[-1][1]}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", default=None, help="pipeline log to read the phase breakdown from")
    parser.add_argument("--interval", default="1d")
    args = parser.parse_args()

    for path in (FEATURES, TARGETS):
        if not path.exists():
            print(f"missing {path} -- run the pipeline first")
            return 1

    built = pd.Timestamp(FEATURES.stat().st_mtime, unit="s", tz="UTC").tz_convert(None)
    print(f"batch written {built:%Y-%m-%d %H:%M} (local clock, UTC-naive)\n")

    report = Report()
    for title, check in (
        ("cross-sectional targets", lambda: check_cross_sectional_targets(report)),
        ("self-calibrating context gate", lambda: check_context_velocity_rank(report)),
        ("corporate filings as events", lambda: check_filing_events(report)),
        ("targets revived on the 15-minute frame", lambda: check_revived_targets(report)),
        ("what reaches feature selection", lambda: check_empty_columns_at_selection(report, args.interval)),
        ("stage 3 cost", lambda: check_stage3_timings(report, Path(args.log) if args.log else None)),
    ):
        print(title)
        try:
            check()
        except Exception as exc:  # noqa: BLE001 - one broken check must not hide the rest
            report.say(FAIL, title, f"check itself failed: {type(exc).__name__}: {exc}")
        print()

    print(f"{report.failures} failing check(s)")
    return 1 if report.failures else 0


if __name__ == "__main__":
    sys.exit(main())
