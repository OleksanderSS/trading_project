"""Why nothing was promoted -- read from the artifact, not from a log.

Stage 4 used to log its refusals and keep none of them, so answering "why has
no return target ever produced a champion" meant finding the run's log and
parsing 446 lines out of it. That worked once, because the log happened to
still exist. The stage now writes `data/results/gate_refusals_*.parquet`, and
this reads it.

The distinction the report is built around is the one that matters:

    no edge     a model was trained, judged, and was not good enough. The
                target may simply not be predictable, or the features may not
                carry it.
    no data     there was never enough to train or judge on. This says nothing
                about predictability at all.

Reading the second as the first is how a data problem gets mistaken for a
verdict about the market. On the batch that was parsed by hand, 342 of 446
refusals were "does not beat the naive baseline" and 24 were "too few events"
-- so that run's answer was "no edge", but only because the split was
counted.

    python scripts/diagnostics/gate_refusal_report.py
    python scripts/diagnostics/gate_refusal_report.py --file data/results/gate_refusals_20260822_193000.parquet
    python scripts/diagnostics/gate_refusal_report.py --target target_relative_return_1d
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

RESULTS = Path("data/results")

#: Substrings that place a refusal in a category. Order matters: the first
#: match wins, and "no data" is checked before "no edge" because a refusal can
#: mention both.
CATEGORIES = (
    ("no data", ("training never ran", "too few", "insufficient", "not enough",
                 "no usable", "holdout_events")),
    ("not promoted by policy", ("measured but not promoted",)),
    ("unstable across folds", ("fold", "stability", "only some")),
    ("loses to one feature", ("single feature", "one feature", "direct")),
    ("no edge vs naive", ("naive", "baseline")),
)


def _categorise(reason: str) -> str:
    lowered = (reason or "").lower()
    for name, needles in CATEGORIES:
        if any(needle in lowered for needle in needles):
            return name
    return "other"


def _latest(directory: Path) -> Path | None:
    candidates = sorted(directory.glob("gate_refusals_*.parquet"))
    return candidates[-1] if candidates else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=Path, default=None,
                        help="a specific artifact; default is the newest")
    parser.add_argument("--target", default=None,
                        help="show every refusal for one target")
    parser.add_argument("--top", type=int, default=15)
    args = parser.parse_args()

    path = args.file or _latest(RESULTS)
    if path is None or not path.exists():
        print(f"{path} does not exist." if args.file
              else f"No gate_refusals_*.parquet in {RESULTS}.")
        print("Stage 4 writes one at the end of a training run; if a run has")
        print("finished and there is no file, EVERY context produced a")
        print("champion -- which would be the first time.")
        return 1

    frame = pd.read_parquet(path)
    print(f"{path.name}: {len(frame)} refusals, "
          f"{frame['target'].nunique()} targets, "
          f"{frame['context'].nunique()} contexts\n")

    frame["category"] = frame["reasons"].map(_categorise)

    print("=== what kind of failure ===")
    counts = frame["category"].value_counts()
    for name, count in counts.items():
        print(f"  {name:24s} {count:5d}   {count / len(frame):6.1%}")
    no_data = int(counts.get("no data", 0))
    print()
    if no_data:
        print(f"  {no_data} of {len(frame)} ({no_data / len(frame):.0%}) say nothing "
              "about predictability:")
        print("  there was not enough to judge. The rest are verdicts on skill.")
    else:
        print("  Every refusal is a verdict on skill; none is a data shortage.")
    print()

    print("=== by target ===")
    grouped = frame.groupby("target")
    rows = []
    for name, part in grouped:
        gaps = (part["holdout_score"] - part["baseline_score"]).dropna()
        rows.append({
            "target": name,
            "refusals": len(part),
            "no data": int((part["category"] == "no data").sum()),
            # Empty when nothing in this group was ever scored -- which is the
            # whole point of the "no data" column next to it, so it prints as a
            # dash rather than as a number pandas would have to invent.
            "median gap": gaps.median() if len(gaps) else float("nan"),
        })
    summary = pd.DataFrame(rows).sort_values("refusals", ascending=False)
    print(f"  {'target':38s} {'refusals':>9s} {'no data':>8s} {'median gap':>11s}")
    for _, row in summary.head(args.top).iterrows():
        gap = row["median gap"]
        gap_text = "        —  " if pd.isna(gap) else f"{gap:>+11.4f}"
        print(f"  {row['target'][:38]:38s} {row['refusals']:9d} "
              f"{row['no data']:8d} {gap_text}")
    if len(summary) > args.top:
        print(f"  ... and {len(summary) - args.top} more targets")
    print()
    print("  A negative gap is the model scoring BELOW the naive baseline on")
    print("  the holdout. The scale differs by target type, so compare within")
    print("  a row, never down a column.")

    if args.target:
        part = frame[frame["target"] == args.target]
        print(f"\n=== every refusal for {args.target} ===")
        if part.empty:
            print("  None. Either it was promoted somewhere, or it was never")
            print("  trained -- the champion artifact says which.")
        else:
            for _, row in part.iterrows():
                print(f"  {row['context']}")
                print(f"      {row['reasons']}")
                if pd.notna(row.get("holdout_score")):
                    counts = " ".join(
                        f"{label} {int(value)}" if pd.notna(value) else f"{label} —"
                        for label, value in (("rows", row.get("holdout_rows")),
                                             ("events", row.get("holdout_events")))
                    )
                    print(f"      holdout {row['holdout_score']:.4f}  "
                          f"baseline {row['baseline_score']:.4f}  {counts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
