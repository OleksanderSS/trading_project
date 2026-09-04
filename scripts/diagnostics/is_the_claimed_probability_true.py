"""When the model says 60%, does it happen 60% of the time?

REGISTER #133: "there is no calibration report -- only a calibrator".
`confidence_calibrator.py` computes Brier and ECE, and it is called only from
the prediction stage, where it FITS a number in flight. Whether that number
can be believed was never measured, and a bet-sizing rule that leans on an
unmeasured probability is leaning on nothing.

The data was already there. `data/results/holdout_predictions_*.parquet`
carries `probability` beside `actual` for 734,286 holdout rows, and the
probability is the model's own output from the trainer -- the confidence
calibrator never touches it. So this is an out-of-sample measurement, not the
calibrator grading itself.

WHAT IS REPORTED, and why each column is here:

    n           per bucket. An ECE computed over buckets holding twelve rows
                is a number about twelve rows. Every claim here carries its
                own count.

    claimed     the mean probability the model asserted in that bucket.

    actual      how often it happened.

    se          the standard error of `actual`, sqrt(p(1-p)/n). "Claimed 60%,
                actual 55%" means nothing until it is read against this.

    Brier       mean squared error of the probability. Lower is better, and
                it is reported against the NAIVE OPPONENT below rather than
                alone.

    base rate   what always predicting the base rate would score. A model
                whose Brier is worse than that is not merely uncalibrated --
                its probabilities carry less than no information, and that
                comparison is the one this project puts first everywhere else.

    ECE         mean |claimed - actual| weighted by bucket size. Reported
                last, because on its own it hides which end is wrong.

Only binary targets are measured: calibration is a statement about the
frequency of an event, and there is no event to count on a return target.

    python scripts/diagnostics/is_the_claimed_probability_true.py
    python scripts/diagnostics/is_the_claimed_probability_true.py --buckets 20
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

RESULTS = PROJECT_ROOT / "data" / "results"

#: A bucket thinner than this says nothing, and saying so is the point.
MIN_BUCKET = 100


def _latest() -> Path:
    files = sorted(glob.glob(str(RESULTS / "holdout_predictions_*.parquet")))
    if not files:
        raise SystemExit(f"no holdout predictions under {RESULTS}")
    return Path(files[-1])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", default=None)
    parser.add_argument("--buckets", type=int, default=10)
    args = parser.parse_args()

    path = Path(args.path) if args.path else _latest()
    frame = pd.read_parquet(path, columns=["target", "model_type",
                                           "probability", "actual"])
    print(f"{path.name}: {len(frame):,} holdout rows\n")

    frame = frame.dropna(subset=["probability", "actual"])
    binary = [
        name for name, group in frame.groupby("target", sort=False)
        if set(pd.unique(group["actual"].astype(float))) <= {0.0, 1.0}
    ]
    skipped = sorted(set(frame["target"]) - set(binary))
    if skipped:
        print("not measurable -- calibration counts events, and these have "
              "none:")
        for name in skipped:
            print(f"  {name}")
        print()

    if not binary:
        print("no binary target in this artifact; nothing to calibrate.")
        return 1

    for target in sorted(binary):
        group = frame[frame["target"] == target]
        claimed = group["probability"].to_numpy(dtype=float)
        actual = group["actual"].to_numpy(dtype=float)
        base = float(actual.mean())

        brier = float(np.mean((claimed - actual) ** 2))
        brier_base = float(np.mean((base - actual) ** 2))

        print(f"=== {target}   n={len(group):,}   base rate {base:.3f}")
        header = (f"{'claimed':>12}{'n':>9}{'mean claimed':>15}"
                  f"{'actual':>10}{'se':>9}{'gap/se':>9}")
        print(header)
        print("-" * len(header))

        edges = np.linspace(0.0, 1.0, args.buckets + 1)
        index = np.clip(np.digitize(claimed, edges[1:-1]), 0, args.buckets - 1)
        weighted_gap = 0.0
        for bucket in range(args.buckets):
            mask = index == bucket
            n = int(mask.sum())
            if n == 0:
                continue
            mean_claimed = float(claimed[mask].mean())
            observed = float(actual[mask].mean())
            se = float(np.sqrt(max(observed * (1 - observed), 1e-12) / n))
            gap = observed - mean_claimed
            weighted_gap += n * abs(gap)
            thin = "  (thin)" if n < MIN_BUCKET else ""
            print(f"{edges[bucket]:>7.1f}-{edges[bucket + 1]:<4.1f}{n:>9,}"
                  f"{mean_claimed:>15.3f}{observed:>10.3f}{se:>9.3f}"
                  f"{gap / se if se else float('nan'):>9.1f}{thin}")

        ece = weighted_gap / len(group)
        verdict = ("BETTER than always predicting the base rate"
                   if brier < brier_base else
                   "WORSE than always predicting the base rate")
        print(f"\n  Brier {brier:.4f}   base-rate Brier {brier_base:.4f}   "
              f"-> {verdict}")
        print(f"  ECE   {ece:.4f}\n")

    print("Read `gap/se` first: a gap of two standard errors or more is a real")
    print("miscalibration; anything smaller is the bucket's own noise. And a")
    print("Brier worse than the base rate means the probabilities carry less")
    print("than no information, whatever the ECE says.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
