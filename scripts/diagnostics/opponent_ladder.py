"""What a candidate must beat before it is called a finding.

Written 2026-08-30, the day the pipeline produced its first non-empty result
and it turned out to be the closing bell.

`target_hourly_volume_spike_1h` won its arena with F1 0.6656 against a naive
constant of 0.1367 -- 4.9x the bar the gate actually checks. It also beat a
lag-1 repeat (0.2200). It died on the fourth opponent: the share of positives
by time of day is 96.5% in the 18:45 UTC slot and 4-6% everywhere else,
because the following hour holds the New York closing auction. A rule that
reads nothing but the clock scores 0.7474 -- better than the champion.

The opponents are ordered by strength, and each is computed from data already
in hand. Nothing here needs a new source; it needs a comparison that was
missing.

    1  base rate          how often the event happens at all
    2  best constant      the best answer that knows nothing about the row
    3  lag of the target  repeating the previous value
    4  the clock          time of day, day of week -- deterministic seasonality
    5  source lag         when the target derives from a column the model sees

A candidate that clears them all is worth testing properly. One that clears
only the first two is what this pipeline used to promote.

    python scripts/diagnostics/opponent_ladder.py --target target_hourly_up_1h
    python scripts/diagnostics/opponent_ladder.py --interval 1d --all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.pipeline.sealed_period import seal_start_for  # noqa: E402

TARGETS = Path("data/colab/accumulated/main_database/targets.parquet")


def target_horizons() -> dict[str, int]:
    """How far ahead each target looks, in bars.

    The lag opponent has to respect this. A forward-looking label at t-1 is
    not known at t: `target_return_5d` at t-1 covers t-1 to t+4, so using it
    to predict t needs four days of the future. Measured 2026-08-30 on the
    daily frame, that illegal opponent scored 0.8053 against the honest
    lag-5's 0.5825 -- and it was one step from disqualifying champions on the
    strength of information no model is allowed to have.

    The last label that has actually resolved by t is the one at t-h, where h
    is the target's own shift.
    """
    import yaml
    path = Path("src/config/targets.yaml")
    if not path.exists():
        return {}
    config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    out: dict[str, int] = {}
    for name, spec in (config.get("targets") or {}).items():
        params = (spec or {}).get("params") or {}
        shift = params.get("shift")
        if isinstance(shift, (int, float)):
            out[str(name)] = max(1, int(abs(shift)))
    return out


def _f1(actual: np.ndarray, predicted: np.ndarray) -> float:
    """F1 on the positive class, which is what the trainers score."""
    true_positive = float(((actual == 1) & (predicted == 1)).sum())
    if true_positive == 0:
        return 0.0
    precision = true_positive / max(float((predicted == 1).sum()), 1.0)
    recall = true_positive / max(float((actual == 1).sum()), 1.0)
    return 2 * precision * recall / (precision + recall)


def ladder(frame: pd.DataFrame, target: str, horizon: int = 1) -> dict[str, float]:
    values = pd.to_numeric(frame[target], errors="coerce")
    usable = values.notna()

    # Refuse a target that is not already binary, instead of coercing it.
    #
    # `(values > 0).astype(int)` was applied to EVERY target here, so a
    # regression target was silently replaced by "did it come out positive"
    # -- a different question from the one the pipeline asks, scored with a
    # different metric, and then quoted beside the pipeline's own numbers.
    #
    # It produced two wrong register entries on 2026-08-30. #174 recorded
    # `target_intraday_volatility_15m` as "declared binary, base rate 1.000,
    # degenerate": it is declared `type: regression`, it is the mean
    # (high-low)/close over the next three bars, and a RANGE IS ALWAYS
    # POSITIVE -- so the 1.000 was this line, not the target. #173 compared a
    # champion's R2 against an F1 computed on the coerced surrogate, which is
    # two metrics with one comparison sign between them.
    #
    # Not repaired by adding regression rungs here: since 2026-08-31 the
    # promotion gate runs the whole ladder itself, on the declared task type
    # and the governing metric, and a second implementation is how this
    # project has repeatedly ended up with two answers to one question.
    distinct = set(pd.unique(values[usable].to_numpy()))
    if not distinct <= {0.0, 1.0, 0, 1, True, False}:
        raise ValueError(
            f"{target} is not binary ({len(distinct)} distinct values); this "
            f"script's rungs are F1-based and coercing it with `> 0` measures "
            f"a different target. Regression targets are judged by the "
            f"promotion gate, which scores the same ladder on R2."
        )

    actual = values[usable].astype(int).to_numpy()
    stamps = frame.loc[usable, "_time"]
    names = frame.loc[usable, "ticker"]
    if actual.size == 0 or actual.sum() == 0:
        return {}

    rate = float(actual.mean())
    rungs = {
        "base rate": rate,
        "always-one": 2 * rate / (1 + rate),
    }

    # The honest lag is the target's own horizon; anything shorter uses a
    # label that has not resolved yet -- see `target_horizons`.
    series = pd.Series(actual, index=names.index)
    repeated = series.groupby(names).shift(horizon).fillna(0).astype(int)
    rungs["lag-h"] = _f1(actual, repeated.to_numpy())
    if horizon != 1:
        illegal = series.groupby(names).shift(1).fillna(0).astype(int)
        rungs["lag-1 (illegal)"] = _f1(actual, illegal.to_numpy())

    # The clock: positives per intraday slot, thresholded at whatever helps
    # most. Deliberately generous -- the point is to find out how much of the
    # target a calendar already explains.
    slot = (stamps.dt.hour * 4 + stamps.dt.minute // 15).to_numpy()
    by_slot = pd.Series(actual).groupby(slot).mean()
    best = 0.0
    for threshold in np.arange(0.02, 0.95, 0.01):
        predicted = (pd.Series(slot).map(by_slot > threshold)
                     .fillna(False).astype(int).to_numpy())
        best = max(best, _f1(actual, predicted))
    rungs["the clock"] = best

    weekday = stamps.dt.dayofweek.to_numpy()
    by_day = pd.Series(actual).groupby(weekday).mean()
    best_day = 0.0
    for threshold in np.arange(0.02, 0.95, 0.01):
        predicted = (pd.Series(weekday).map(by_day > threshold)
                     .fillna(False).astype(int).to_numpy())
        best_day = max(best_day, _f1(actual, predicted))
    rungs["day of week"] = best_day
    return rungs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interval", default="15m")
    parser.add_argument("--target", default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--champion", type=float, default=None,
                        help="the candidate's F1, printed beside the opponents")
    args = parser.parse_args()

    if not TARGETS.exists():
        print(f"no {TARGETS}")
        return 2

    names = pq.ParquetFile(TARGETS).schema_arrow.names
    wanted = ([args.target] if args.target
              else [c for c in names if c.startswith("target_")])

    frame = pd.read_parquet(
        TARGETS, columns=["datetime", "ticker", "interval", *wanted]
    )
    frame = frame[frame["interval"].astype(str) == args.interval]
    frame["_time"] = pd.to_datetime(frame["datetime"], errors="coerce", utc=True)
    # Per frame, because an absolute date swallows a short one whole.
    seal = seal_start_for(frame["_time"])
    before = len(frame)
    frame = frame[frame["_time"] < seal]
    print(f"sealed from {seal:%Y-%m-%d} onward for the {args.interval} frame")
    print(f"  {before - len(frame):,} rows withheld; {len(frame):,} remain")
    print()

    horizons = target_horizons()
    header = (f"{'target':34s} {'h':>3s} {'base':>6s} {'const':>7s} "
              f"{'lag-h':>7s} {'clock':>7s} {'weekday':>7s}  strongest (legal)")
    print(header)
    print("-" * len(header))
    for target in wanted:
        rungs = ladder(frame, target, horizons.get(target, 1))
        if not rungs:
            continue
        h = horizons.get(target, 1)
        legal = {k: v for k, v in rungs.items()
                 if k != "base rate" and "illegal" not in k}
        strongest = max(legal, key=legal.get)
        print(f"{target[:34]:34s} {h:3d} {rungs['base rate']:6.3f} "
              f"{rungs['always-one']:7.4f} {rungs['lag-h']:7.4f} "
              f"{rungs['the clock']:7.4f} {rungs['day of week']:7.4f}  "
              f"{strongest} ({legal[strongest]:.4f})")
        if args.champion is not None:
            verdict = ("BEATS every opponent"
                       if args.champion > opponents[strongest]
                       else f"LOSES to {strongest}")
            print(f"{'  candidate':38s} {args.champion:>39.4f}  {verdict}")
    print()
    print("A candidate must beat the STRONGEST opponent, not the first one.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
