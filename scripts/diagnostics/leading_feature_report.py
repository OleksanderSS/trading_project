"""Is this feature leading, is it usable, and would it survive being traded?

Run three times by hand on 2026-08-23 and it caught three different lies, so it
is a tool now rather than a scratch script. Each column below exists because
something passed the other checks and failed this one.

    varies        Does the feature differ BETWEEN names on the same date? A
                  market-wide series does not: `cftc_*` has cross-sectional
                  variation on 0.0% of dates, so in a ranking it contributes
                  exactly zero however good its correlation looks. Such a
                  series is not useless -- it enters as an interaction with a
                  per-name sensitivity, never as a column.

    ic_out        Rank correlation with the target on a HOLDOUT split by time.
                  Chosen on the train slice, measured after it. In-sample IC
                  measures memory, not skill.

    kept_sign     Does the direction survive the split? Under the null this is
                  a coin flip. It is the cheapest check that separates a
                  relationship from a coincidence.

    ic_within     The same correlation after each ticker's own mean is removed.
                  This is the one that matters most and is easiest to skip. A
                  cross-sectional book built without it held NVDA on 100% of
                  days against SPY on 100% of days and "returned" 34.9% a year
                  -- which is not a signal but one bet, discovered after the
                  fact. If ic_within collapses toward zero, the feature is not
                  predicting WHEN, it is labelling WHICH NAME.

    history       How many rows the feature actually has. Attention was
                  collected 30 days deep against a frame spanning decades: the
                  column existed, reported success, and could never be trained
                  on. Depth is checked before anything else is believed.

Nothing here says a feature is profitable. It says whether a feature is worth
the cost of testing properly.

    python scripts/diagnostics/leading_feature_report.py
    python scripts/diagnostics/leading_feature_report.py --target target_return_5d --top 40
    python scripts/diagnostics/leading_feature_report.py --features wiki_views filing_count_30d_1d
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import spearmanr

# The diagnostics run as scripts, not as part of the package, so the
# project root has to be on the path before src/ can be imported.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.pipeline.sealed_period import SEAL_START, describe

FEATURES = Path("data/colab/accumulated/main_database/features.parquet")
TARGETS = Path("data/colab/accumulated/main_database/targets.parquet")

DEFAULT_TARGET = "target_relative_return_5d"
TRAIN_FRACTION = 0.70
BLOCK = 250

#: Below this a correlation is being read off too few rows to mean anything.
MIN_ROWS = 500

#: A feature covering less of the frame than this cannot be trained on, whatever
#: its correlation says. Attention sat at 0.4% before 2026-08-23.
THIN_COVERAGE = 0.05


def _daily_targets(target: str) -> pd.DataFrame:
    frame = pd.read_parquet(TARGETS, columns=["datetime", "ticker", "interval", target])
    frame = frame[frame["interval"].astype(str).eq("1d")].reset_index(drop=True)
    frame["datetime"] = pd.to_datetime(frame["datetime"], errors="coerce", utc=True)
    return frame


def _safe_ic(values: np.ndarray, outcome: np.ndarray, mask: np.ndarray) -> float:
    return _safe_ic_p(values, outcome, mask)[0]


def _safe_ic_p(values: np.ndarray, outcome: np.ndarray,
               mask: np.ndarray) -> tuple[float, float]:
    """The correlation and the p-value that goes with it.

    The p-value was computed and thrown away. It is the only thing that lets
    the report say how many of its survivors the null would have produced:
    459 features were screened on 2026-08-29 and 22 came back "worth testing
    properly" -- while a 5% rate on 459 tests yields about 23 by chance. The
    list was exactly the size of the noise it was supposed to beat.
    """
    usable = mask & np.isfinite(values) & np.isfinite(outcome)
    if usable.sum() < MIN_ROWS:
        return float("nan"), float("nan")
    if np.nanstd(values[usable]) == 0 or np.nanstd(outcome[usable]) == 0:
        return float("nan"), float("nan")
    result = spearmanr(values[usable], outcome[usable])
    return float(result.statistic), float(result.pvalue)


def benjamini_hochberg(pvalues: np.ndarray, alpha: float = 0.05) -> float:
    """Largest p-value that still passes at a 5% false-discovery rate."""
    finite = np.sort(pvalues[np.isfinite(pvalues)])
    if finite.size == 0:
        return 0.0
    ranks = np.arange(1, finite.size + 1)
    passing = finite <= alpha * ranks / finite.size
    return float(finite[passing].max()) if passing.any() else 0.0


def per_date_ic(values: np.ndarray, outcome: np.ndarray, mask: np.ndarray,
                dates: pd.Series, min_names: int = 10) -> tuple[float, float, int]:
    """Cross-sectional IC per date, then a t-test over those dates.

    Pooling 210,000 holdout rows into one Spearman and reading its p-value
    treats every row as an independent observation. They are nothing of the
    kind: 110 names move together on a date, and a 5-day forward return
    overlaps its neighbour by four days. Measured that way, IC 0.01 came back
    at p about 4e-6 and 167 of 447 features cleared a Benjamini-Hochberg
    threshold -- the correction did not bind, because the inputs were wrong
    before the correction ever ran.

    The unit of observation is the DATE. Rank across names inside each date,
    correlate, and test the resulting series of daily coefficients against
    zero. n becomes the number of dates, the cross-sectional dependence is
    absorbed into each date's single number, and the surviving overlap shows
    up as autocorrelation -- still optimistic, but by a factor, not by orders
    of magnitude.

    Returns (mean IC, t-statistic, dates used).
    """
    usable = mask & np.isfinite(values) & np.isfinite(outcome)
    if usable.sum() < MIN_ROWS:
        return float("nan"), float("nan"), 0
    frame = pd.DataFrame({
        "v": values[usable], "o": outcome[usable],
        "d": pd.Series(dates).to_numpy()[usable],
    })
    counts = frame.groupby("d")["v"].transform("size")
    frame = frame[counts >= min_names]
    if frame.empty:
        return float("nan"), float("nan"), 0

    grouped = frame.groupby("d")
    ranks_v = grouped["v"].rank()
    ranks_o = grouped["o"].rank()
    work = pd.DataFrame({"d": frame["d"], "rv": ranks_v, "ro": ranks_o})
    stats = work.groupby("d").agg(
        n=("rv", "size"), mv=("rv", "mean"), mo=("ro", "mean"),
        sv=("rv", "std"), so=("ro", "std"),
    )
    work["prod"] = work["rv"] * work["ro"]
    stats["mp"] = work.groupby("d")["prod"].mean()
    cov = (stats["mp"] - stats["mv"] * stats["mo"]) * stats["n"] / (stats["n"] - 1)
    daily = cov / (stats["sv"] * stats["so"])
    daily = daily.replace([np.inf, -np.inf], np.nan).dropna()
    if len(daily) < 30:
        return float("nan"), float("nan"), int(len(daily))
    mean = float(daily.mean())
    spread = float(daily.std(ddof=1))
    t = mean / (spread / np.sqrt(len(daily))) if spread > 0 else float("nan")
    per_date_ic.last_series = daily
    return mean, float(t), int(len(daily))


def _t_of(daily: pd.Series) -> float:
    """t against zero for a series of daily coefficients."""
    if daily is None or len(daily) < 20:
        return float("nan")
    spread = float(daily.std(ddof=1))
    if spread <= 0:
        return float("nan")
    return float(daily.mean() / (spread / np.sqrt(len(daily))))


def stability(daily: pd.Series, blocks: int = 4) -> tuple[int, float]:
    """Does the coefficient hold up in the LATEST stretch, or is it a fossil?

    A pooled t hides decay, and decay is the commonest way an effect is both
    real and useless. Measured on `insider_net_value_30d` on 2026-08-29: the
    pooled t of -6.26 was the strongest number in the batch, and splitting it
    by period gave -2.74, -3.01, **-6.01 across the spring of 2020**, -1.88,
    and **-0.08 over the final twenty months**. The sign never flipped; the
    effect simply stopped. Traded on the pooled number, that is a strategy
    fitted to a market that no longer exists.

    The daily series is already computed, so this costs nothing extra.
    Returns (blocks agreeing with the overall sign, t of the last block).
    """
    if daily is None or len(daily) < blocks * 20:
        return 0, float("nan")
    ordered = daily.sort_index()
    parts = np.array_split(ordered.to_numpy(), blocks)
    overall = np.sign(ordered.mean())
    agree = sum(1 for part in parts
                if len(part) and np.sign(np.mean(part)) == overall)
    return int(agree), _t_of(pd.Series(parts[-1]))


def _demean(values: np.ndarray, keys: np.ndarray) -> np.ndarray:
    series = pd.Series(values)
    return (series - series.groupby(pd.Series(keys)).transform("mean")).to_numpy()


def _examine(name: str, values: np.ndarray, book: dict) -> dict:
    outcome, is_train = book["outcome"], book["is_train"]
    finite = np.isfinite(values)
    coverage = finite.mean()

    per_date_spread = pd.Series(values).groupby(book["dates"]).std()
    varies = float((per_date_spread.fillna(0) > 1e-12).mean())

    ic_in = _safe_ic(values, outcome, is_train)
    ic_out, _pooled = _safe_ic_p(values, outcome, ~is_train)
    ic_daily, t_daily, n_dates = per_date_ic(
        values, outcome, ~is_train, book["dates"]
    )
    blocks_agree, t_recent = stability(getattr(per_date_ic, "last_series", None))
    # The p-value that feeds the correction comes from the DATE series, not
    # from the pooled rows. See per_date_ic for why the pooled one is unusable.
    from scipy.stats import t as _t
    p_out = (float(2 * _t.sf(abs(t_daily), df=max(n_dates - 1, 1)))
             if np.isfinite(t_daily) else float("nan"))
    within = _safe_ic(_demean(values, book["tickers"]),
                      book["outcome_demeaned"], ~is_train)

    return {
        "feature": name,
        "history": int(finite.sum()),
        "coverage": coverage,
        "varies": varies,
        "ic_in": ic_in,
        "ic_out": ic_out,
        "ic_daily": ic_daily,
        "t_daily": t_daily,
        "n_dates": n_dates,
        "blocks_agree": blocks_agree,
        "t_recent": t_recent,
        "p_out": p_out,
        "kept_sign": bool(np.isfinite(ic_in) and np.isfinite(ic_out)
                          and np.sign(ic_in) == np.sign(ic_out)),
        "ic_within": within,
    }


def _verdict(row: pd.Series) -> str:
    if row["coverage"] < THIN_COVERAGE:
        return "too thin to judge"
    if row["varies"] < 0.05:
        return "market-wide: use as interaction"
    if not np.isfinite(row["ic_out"]):
        return "not measurable"
    if not row["kept_sign"]:
        return "sign flipped out of sample"
    if abs(row["ic_within"]) < abs(row["ic_out"]) * 0.34:
        return "labels the name, not the moment"
    if not row.get("passes_fdr", True):
        return "inside the noise for this many tests"
    recent = row.get("t_recent", float("nan"))
    if np.isfinite(recent) and abs(recent) < 1.0:
        return "faded: held once, gone in the latest quarter"
    if (np.isfinite(recent) and np.isfinite(row.get("t_daily", np.nan))
            and np.sign(recent) != np.sign(row["t_daily"])):
        return "reversed in the latest quarter"
    if abs(row["ic_out"]) < 0.01:
        return "survives, but tiny"
    return "survives, worth testing"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default=DEFAULT_TARGET)
    parser.add_argument("--features", nargs="*", default=None,
                        help="specific columns; default is every usable one")
    parser.add_argument("--top", type=int, default=25)
    args = parser.parse_args()

    if not FEATURES.exists() or not TARGETS.exists():
        print("No batch on disk; run --mode prepare first.")
        return 1

    targets = _daily_targets(args.target)
    if targets[args.target].notna().sum() < MIN_ROWS:
        print(f"{args.target} has almost no values on the daily frame.")
        return 1

    # Nothing at or after the seal is read. See src/pipeline/sealed_period.
    unsealed = (targets["datetime"] < SEAL_START).to_numpy()
    sealed_rows = int((~unsealed).sum())
    targets = targets.loc[unsealed].reset_index(drop=True)
    print(describe())
    print(f"  {sealed_rows:,} daily rows withheld; {len(targets):,} remain.")
    print()

    cut = targets["datetime"].quantile(TRAIN_FRACTION)
    outcome = targets[args.target].to_numpy(dtype=float)
    book = {
        "outcome": outcome,
        "outcome_demeaned": _demean(outcome, targets["ticker"].to_numpy()),
        "is_train": (targets["datetime"] <= cut).to_numpy(),
        "dates": targets["datetime"],
        "tickers": targets["ticker"].to_numpy(),
    }
    print(f"target {args.target} | daily rows {len(targets):,} | split {cut.date()} "
          f"| {targets['ticker'].nunique()} names\n")

    mask = pd.read_parquet(FEATURES, columns=["interval"])
    mask = mask["interval"].astype(str).eq("1d").to_numpy()
    # Restrict the True positions to the same unsealed rows the targets kept.
    # Both files are written by one union writer in the same order per
    # timeframe, so position i among the 1d rows means the same bar in each.
    mask[mask] = unsealed

    wanted = args.features or [
        name for name in pq.ParquetFile(FEATURES).schema_arrow.names
        if name not in {"datetime", "ticker", "interval"}
        and not name.startswith("target_")
    ]

    rows = []
    for start in range(0, len(wanted), BLOCK):
        columns = wanted[start:start + BLOCK]
        try:
            block = pd.read_parquet(FEATURES, columns=columns)
        except (OSError, ValueError):
            continue
        block = block.loc[mask].reset_index(drop=True)
        for name in columns:
            if name not in block.columns:
                continue
            values = pd.to_numeric(block[name], errors="coerce").to_numpy(dtype=float)
            if np.isfinite(values).sum() == 0:
                continue
            rows.append(_examine(name, values, book))
        del block

    if not rows:
        print("None of the requested features exist on the daily frame.")
        return 1

    report = pd.DataFrame(rows)

    # The correction is applied over everything that was MEASURABLE, because
    # that is how many tests were actually run. Restricting it to the ones
    # that already look good is the same mistake in a different place.
    threshold = benjamini_hochberg(report["p_out"].to_numpy())
    report["passes_fdr"] = report["p_out"].le(threshold).fillna(False)
    report["verdict"] = report.apply(_verdict, axis=1)
    report = report.reindex(
        report["ic_out"].abs().sort_values(ascending=False, na_position="last").index
    )

    print(f"{'feature':34s} {'cov':>5s} {'varies':>7s} {'ic_out':>8s} "
          f"{'within':>8s} {'ic/date':>8s} {'t':>6s} {'dates':>6s}  verdict")
    print("-" * 118)
    for _, row in report.head(args.top).iterrows():
        ic_out = "      . " if pd.isna(row["ic_out"]) else f"{row['ic_out']:+8.4f}"
        within = "      . " if pd.isna(row["ic_within"]) else f"{row['ic_within']:+8.4f}"
        icd = "      . " if pd.isna(row["ic_daily"]) else f"{row['ic_daily']:+8.4f}"
        tst = "     . " if pd.isna(row["t_daily"]) else f"{row['t_daily']:+6.2f}"
        print(f"{row['feature'][:34]:34s} {row['coverage']:4.0%} "
              f"{row['varies']:6.0%} {ic_out} {within} {icd} {tst} "
              f"{row['n_dates']:6,}  {row['verdict']}")

    print()
    print("=== how many features fall into each verdict ===")
    for verdict, count in report["verdict"].value_counts().items():
        print(f"  {verdict:34s} {count:5d}")
    print()
    measurable = int(report["p_out"].notna().sum())
    passing = int(report["passes_fdr"].sum())
    print(f"{measurable:,} features carried a measurable correlation, so that is")
    print("how many hypotheses were tested.")
    print(f"  Benjamini-Hochberg at 5% FDR: p <= {threshold:.2e}, {passing} pass.")
    print(f"  That controls the share of FALSE ones among those {passing} at 5%")
    print(f"  -- roughly {passing * 0.05:.0f} of them, not all of them.")
    print(f"  For contrast only: a naive p<0.05 cutoff applied {measurable:,}")
    print(f"  times would hand back about {measurable * 0.05:.0f} features from pure noise.")
    print()
    survivors = report[report["verdict"] == "survives, worth testing"]

    # The survivors are the result, so they are printed in full -- always.
    # The table above is ranked by |ic_out| and cut at --top, and a feature
    # can survive every check while sitting below that cut: on 2026-08-29
    # three of the seven were visible and four were not, so the finding had
    # to be reconstructed by hand from a list that did not contain it.
    if len(survivors):
        print("=== every feature that survived, regardless of the cut above ===")
        for _, row in survivors.iterrows():
            print(f"  {row['feature'][:38]:38s} ic/date {row['ic_daily']:+.4f} "
                  f"t {row['t_daily']:+6.2f} over {row['n_dates']:,} dates, "
                  f"coverage {row['coverage']:.0%}, "
                  f"latest quarter t {row['t_recent']:+.2f}, "
                  f"{row['blocks_agree']}/4 quarters agree")
        print()

    # The measured roles are kept, not just printed.
    #
    # Every run of this report has so far ended in scrollback: the verdicts
    # existed for as long as the terminal did. A role is a measurement, and a
    # measurement that is not written down gets made again -- on 2026-08-29
    # the same features were screened four times in one day because there was
    # nowhere to look up what they had already been found to be.
    #
    # The catalogue is therefore an artifact of this script rather than a
    # document anyone maintains: it is rewritten whenever the measurement is
    # rerun, and it carries the batch and seal it was measured against so a
    # later reader can tell whether it is still about their data.
    # This report is built on the daily frame only; the name says so.
    out = Path("diagnostic_reports") / "feature_roles_1d.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    keep = report.copy()
    keep.insert(0, "measured_on", pd.Timestamp.utcnow().strftime("%Y-%m-%d"))
    keep.insert(1, "target", args.target)
    keep.insert(2, "sealed_from", str(SEAL_START.date()))
    keep.insert(3, "tests_screened", measurable)
    keep.to_csv(out, index=False)
    print(f"roles written to {out} ({len(keep)} features)")
    print()

    print(f"{len(survivors)} of {len(report)} are worth the cost of testing properly.")
    print("None of this says a feature is profitable. It says which ones are")
    print("not already disqualified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
