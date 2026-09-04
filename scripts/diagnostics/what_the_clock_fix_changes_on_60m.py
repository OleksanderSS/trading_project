"""REGISTER #234: measure the #231 fix where it can actually bite.

The clock opponent used to pick its scheme by scoring all three on the HOLDOUT
and keeping the best -- a maximum chosen on the rows it is then compared
against. #231 moved that choice to validation.

Re-measuring R13 afterwards changed nothing and cost nine hours, and the reason
was measurable in five seconds: on a DAILY frame `weekday` and `weekday_hour`
are the same predictor, because the hour is constant, so there is nothing to
choose between. On 60m they are genuinely different -- the three schemes agree
on 48.5%, 63.2% and 57.0% of rows -- so 60m is the only place the fix can
change a number.

This measures the MECHANISM rather than running a five-hour training pass. The
model's score is unaffected by which scheme the opponent picks; what the fix
changes is the OPPONENT's holdout score, and that is computable directly:

    old  choose by holdout score, report that score
    new  choose by validation score, report the chosen scheme's holdout score

The difference between those two numbers is the whole effect of #231, and it
can only be positive -- the old method takes a maximum, so it can never score
lower than a scheme chosen elsewhere.

    python scripts/diagnostics/what_the_clock_fix_changes_on_60m.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.core.logging.logger import ProjectLogger  # noqa: E402
from src.metrics.model.ml_evaluator import MLEvaluator  # noqa: E402
from src.training.batch_trainer import BatchTrainer  # noqa: E402

BATCH = PROJECT_ROOT / "data" / "colab" / "accumulated" / "main_database"
from src.pipeline.sealed_period import SEAL_START  # noqa: E402

#: Imported, never restated. Eight diagnostics each kept their own copy of this
#: date until 2026-09-04. The policy in docs/SEALED_HOLDOUT.md says moving the
#: seal EARLIER is "always safe" -- with eight copies it would have been safe in
#: one file and silently ignored in the other seven, which is the duplication
#: family this codebase's defects come from.
SEALED = SEAL_START


def _trainer() -> BatchTrainer:
    """The real methods without the constructor's database side effect."""
    instance = BatchTrainer.__new__(BatchTrainer)
    instance.config_manager = None
    instance.logger = ProjectLogger.get_logger("ClockFixOn60m")
    instance.evaluator = MLEvaluator()
    return instance


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default="target_hourly_up_1h")
    parser.add_argument("--interval", default="60m")
    args = parser.parse_args()

    ident = ["ticker", "datetime", "interval"]
    targets = pd.read_parquet(BATCH / "targets.parquet",
                              columns=ident + [args.target])
    block = targets[targets["interval"] == args.interval].copy()
    block["datetime"] = pd.to_datetime(block["datetime"], utc=True)
    block = block[block["datetime"] < SEALED].dropna(subset=[args.target])
    block = block.sort_values(["datetime", "ticker"]).reset_index(drop=True)

    n = len(block)
    train_end, val_end = int(n * 0.6), int(n * 0.7)
    print(f"{args.target} on {args.interval}: {n:,} rows, "
          f"{block['ticker'].nunique()} names, "
          f"{block['datetime'].min().date()} -> {block['datetime'].max().date()}")
    print(f"train {train_end:,} | val {val_end - train_end:,} | "
          f"holdout {n - val_end:,}\n")

    def frame_of(lo: int, hi: int) -> pd.DataFrame:
        stamps = pd.DatetimeIndex(block["datetime"].iloc[lo:hi],
                                  name="model_datetime")
        return pd.DataFrame({"f": np.zeros(hi - lo)}, index=stamps)

    y = block[args.target].to_numpy(dtype=float)
    data = {
        "X_train": frame_of(0, train_end),
        "X_val": frame_of(train_end, val_end),
        "X_holdout": frame_of(val_end, n),
        "y_val": y[train_end:val_end],
    }
    y_train, y_holdout = y[:train_end], y[val_end:]

    trainer = _trainer()
    metric, task = "BalancedAccuracy", "classification"

    holdout_schemes = trainer._clock_prediction(data, y_train, True,
                                                split="X_holdout")
    val_schemes = trainer._clock_prediction(data, y_train, True, split="X_val")
    if not holdout_schemes:
        print("no clock scheme is available on this frame; #234 cannot be measured here")
        return 1

    print(f"{'scheme':<16}{'val score':>12}{'holdout score':>15}{'buckets':>9}")
    print("-" * 52)
    holdout_scores, val_scores = {}, {}
    for name in sorted(holdout_schemes):
        prediction, buckets = holdout_schemes[name]
        holdout_scores[name] = float(trainer.evaluator.calculate(
            y_holdout, prediction, task_type=task).get(metric, 0.0))
        v = val_schemes.get(name)
        val_scores[name] = float(trainer.evaluator.calculate(
            data["y_val"], v[0], task_type=task).get(metric, 0.0)) if v else float("nan")
        print(f"{name:<16}{val_scores[name]:>12.4f}{holdout_scores[name]:>15.4f}"
              f"{buckets:>9}")

    old_pick = max(holdout_scores, key=lambda k: holdout_scores[k])
    new_pick, how = trainer._choose_clock_scheme(
        data, y_train, True, task, metric, list(holdout_schemes))

    print("\n" + "=" * 52)
    print(f"OLD, chosen on the holdout : {old_pick:<14} scores "
          f"{holdout_scores[old_pick]:.4f}")
    print(f"NEW, chosen on validation  : {new_pick:<14} scores "
          f"{holdout_scores.get(new_pick, float('nan')):.4f}  ({how})")
    delta = holdout_scores[old_pick] - holdout_scores.get(new_pick, np.nan)
    print(f"\nthe opponent is now WEAKER by {delta:+.4f} balanced accuracy")
    if old_pick == new_pick:
        print("Same scheme either way: on this target the fix is inert, and the")
        print("nine hours R13 cost would have been saved by this five-second check.")
    else:
        print("Different scheme: the old method was handing the opponent a")
        print("maximum chosen on the very rows it was compared against, and on")
        print("this frame that is worth the number above -- every intraday")
        print("verdict before 2026-09-03 was decided against it.")
    print("=" * 52)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
