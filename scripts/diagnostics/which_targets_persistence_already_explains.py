"""Which targets does "the same value, h bars ago" already explain?

Before measuring whether any FEATURE leads a target (ROADMAP 1.2), it is worth
knowing which targets are worth leading. A target that its own lagged value
predicts is not a forecasting problem: a model that scores well on it has
learned the calendar, not the market.

This is not hypothetical here. Seven indicator-prediction targets produced 138
of 354 champions on an earlier batch, and persistence alone explained them:

    target_sma_20_f1        R2 0.9994
    target_ema_20_f1        R2 0.9994
    target_bb_upper_f1      R2 0.9984
    target_macd_hist_f1     R2 0.9264

(`base_trainer._score_naive_baselines`). REGISTER #204 then found the same
shape in a family that is still live -- smoothed targets where lag-1 gives
R2 0.8946 while the model gives -0.0020 -- and it was found BY ACCIDENT, in a
run log. Scan unit P1 (target construction) has never been walked deliberately,
and the ledger's own rule says a unit nobody has looked at is a coverage defect
until shown otherwise.

So: every target in the batch, scored against the one opponent that costs
nothing.

WHAT IS MEASURED, and why it is the honest version:

  * TWO lags, because the pipeline has two answers and they disagree.
    `target_horizon_bars` reads the horizon out of the target's NAME;
    `_get_target_horizon_rows` asks the policy manager, which also counts the
    forward WINDOW. On 2026-09-03 they were measured against each other and
    differ on exactly the targets this script flags:

        target_daily_trend_strength_1d     name says 1, window says 20
        target_daily_momentum_score_1d     name says 1, window says 10
        target_hourly_breakout_1h          name says 1, window says  4

    The first version of this script used the NAME horizon and called the
    result a tautology. That is the same defect #191 fixed inside the gate:
    lagging a multi-bar target by one bar uses a value nobody could know at
    forecast time, so the opponent is an oracle and its score means nothing
    about the target. Both are reported now, because they answer different
    questions -- the name lag is what the GATE actually applies
    (`base_trainer` line 1223 calls `target_horizon_bars`), and the window lag
    is whether the TARGET repeats itself.
  * the lag is taken WITHIN a ticker. On a pooled frame "the row h positions
    back" is a different company, which measures nothing.
  * binary targets are scored with balanced accuracy and continuous ones with
    R2, matching what the gate uses, so the numbers here are comparable with
    the ones in a refusal message.

READING THE OUTPUT: a high score is bad news about the TARGET, not good news
about anything. It says the target repeats itself, so any model fitted to it
is graded against an opponent that already knows the answer.

    python scripts/diagnostics/which_targets_persistence_already_explains.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.pipeline.stages.modeling.walk_forward_validation import (  # noqa: E402
    _get_target_horizon_rows,
)
from src.targets.timeframe_contract import target_horizon_bars  # noqa: E402

BATCH = PROJECT_ROOT / "data" / "colab" / "accumulated" / "main_database"


def _is_binary(values: pd.Series) -> bool:
    unique = pd.unique(values.dropna())
    return len(unique) <= 2 and set(np.asarray(unique, dtype=float)) <= {0.0, 1.0}


def _balanced_accuracy(truth: np.ndarray, guess: np.ndarray) -> float:
    scores = []
    for label in (0.0, 1.0):
        mask = truth == label
        if not mask.any():
            return float("nan")
        scores.append(float((guess[mask] == label).mean()))
    return float(np.mean(scores))


def _r2(truth: np.ndarray, guess: np.ndarray) -> float:
    residual = float(np.sum((truth - guess) ** 2))
    total = float(np.sum((truth - truth.mean()) ** 2))
    return float("nan") if total <= 0 else 1.0 - residual / total


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, default=BATCH)
    args = parser.parse_args()

    path = args.batch / "targets.parquet"
    if not path.exists():
        print(f"no targets at {path}")
        return 1

    frame = pd.read_parquet(path)
    targets = [c for c in frame.columns if c.startswith("target_")]
    frame["datetime"] = pd.to_datetime(frame["datetime"], utc=True)
    frame = frame.sort_values(["ticker", "datetime"])

    print(f"{len(frame):,} rows, {len(targets)} target columns, "
          f"intervals {sorted(frame['interval'].unique())}\n")
    header = (f"{'target':<34}{'tf':>5}{'name':>6}{'window':>8}{'rows':>11}"
              f"{'metric':>16}{'lag=name':>10}{'lag=window':>10}")
    print(header)
    print("-" * len(header))

    rows = []
    for interval in sorted(frame["interval"].unique()):
        block = frame[frame["interval"] == interval]
        for name in targets:
            values = block[name]
            if values.notna().sum() < 500:
                continue
            name_lag = target_horizon_bars(name, interval) or 1
            window_lag = _get_target_horizon_rows(name) or name_lag
            binary = _is_binary(values)
            metric = "BalancedAccuracy" if binary else "R2"

            scored = {}
            for label, lag in (("name", name_lag), ("window", window_lag)):
                lagged = block.groupby("ticker", sort=False)[name].shift(lag)
                usable = values.notna() & lagged.notna()
                if int(usable.sum()) < 500:
                    scored[label] = (float("nan"), 0)
                    continue
                truth = values[usable].to_numpy(dtype=float)
                guess = lagged[usable].to_numpy(dtype=float)
                score = (_balanced_accuracy(truth, guess) if binary
                         else _r2(truth, guess))
                scored[label] = (score, int(usable.sum()))

            rows.append((name, interval, name_lag, window_lag, metric,
                         scored["name"][0], scored["window"][0],
                         scored["window"][1]))
            flag = " <-- lags disagree" if name_lag != window_lag else ""
            print(f"{name:<34}{interval:>5}{name_lag:>6}{window_lag:>8}"
                  f"{scored['window'][1]:>11,}{metric:>16}"
                  f"{scored['name'][0]:>10.4f}{scored['window'][0]:>10.4f}{flag}")

    print("\n" + "=" * len(header))
    # The thresholds are the gate's own reference points, not new ones: a
    # constant scores 0.5 on balanced accuracy and 0.0 on R2, so anything a
    # long way above that is explained before a model is fitted.
    def _high(metric, score):
        return (metric == "R2" and score > 0.5) or (
            metric == "BalancedAccuracy" and score > 0.60)

    tautologies = [r for r in rows if _high(r[4], r[6])]
    oracles = [r for r in rows if r[2] != r[3] and _high(r[4], r[5])
               and not _high(r[4], r[6])]

    if tautologies:
        print(f"{len(tautologies)} target(s) that REPEAT THEMSELVES at their own "
              f"horizon -- a model fitted to these is graded against an "
              f"opponent that already knows the answer:")
        for r in sorted(tautologies, key=lambda r: -r[6]):
            print(f"    {r[0]:<34}{r[1]:>5}  lag-{r[3]}  {r[4]} {r[6]:.4f}")
    else:
        print("No target repeats itself at its own horizon. Where the lag is "
              "chosen honestly, persistence is a real opponent rather than a "
              "symptom.")

    if oracles:
        print(f"\n{len(oracles)} target(s) where THE GATE'S OPPONENT IS AN ORACLE: "
              f"high at the name lag, ordinary at the window lag. "
              f"`base_trainer` builds its persistence baseline with "
              f"`target_horizon_bars`, which reads the NAME, so for these it "
              f"lags by less than the target reaches forward and compares the "
              f"model against a value nobody could know. That refuses real "
              f"edges; it does not promote noise.")
        for r in sorted(oracles, key=lambda r: -r[5]):
            print(f"    {r[0]:<34}{r[1]:>5}  lag-{r[2]} {r[5]:.4f}  vs  "
                  f"lag-{r[3]} {r[6]:.4f}")
    print("=" * len(header))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
