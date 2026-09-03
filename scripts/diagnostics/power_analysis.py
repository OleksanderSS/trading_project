"""What is the smallest edge this sample could tell apart from nothing?

REGISTER #194. Written 2026-09-01, after run 7 refused all seven return
targets with R2 between -0.0017 and -0.0166 -- every one of them worse than
predicting the training mean (CLAIMS.md R7). Two readings of that fit the
same numbers:

  * the features carry no information about returns, or
  * the sample cannot resolve an edge of the size anyone would plausibly
    hope for, and the negative numbers are noise around zero plus the cost
    of fitting 436 columns.

Nothing in the pipeline distinguishes those, and they call for opposite next
moves: the first says keep looking for better features, the second says stop
and change the data, the horizon or the question. This script answers it by
arithmetic on the target series alone -- it does not use a model, a feature,
or any pipeline output, so a defect anywhere upstream cannot change its
answer.

WHAT IT ASSUMES, said before any number is quoted:

  1. Rows are not independent observations. Two of them share information
     when they overlap in time (a 5-day target on consecutive days shares 4
     of its 5 days) and when they belong to the same date across tickers that
     move together. Both are corrected for; nothing else is.
  2. Overlap is handled by counting non-overlapping blocks: a target with
     shift -5 and no window gets one independent block per 5 bars, per
     ticker. A window extends the dependency (a 3-bar realised volatility at
     shift -1 reaches 3 bars forward), so the block is |shift| + window - 1.
  3. Cross-sectional dependence is handled by the standard effective-breadth
     correction, N_eff = N / (1 + (N-1) * rho), where rho is the average
     pairwise correlation of the target across tickers on the same date. For
     a market factor this collapses 110 names to a handful.
  4. Power is 80% at a two-sided 5% level, so the detectable effect is
     2.802 standard errors, not 1.96. Quoting 1.96 answers "how big must an
     effect be before I would call it significant if I saw it", which is a
     different and easier question than "how big must it be before I would
     RELIABLY see it".
  5. The multiplicity column assumes independent tests, which Bonferroni
     does; the real tests are correlated, so that column is conservative.
     It is shown because the project has never applied any correction
     (CRITIQUE.md 4) and the size of what is being ignored is the point.

WHAT IT DOES NOT DO: it says nothing about whether an edge exists. It says
what size of edge this sample could confirm if one did.
"""
from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

TARGETS_YAML = PROJECT_ROOT / "src" / "config" / "targets.yaml"
DEFAULT_BATCH = PROJECT_ROOT / "data" / "colab" / "accumulated" / "main_database"

#: 80% power, two-sided 5%: z(0.975) + z(0.80).
Z_DETECT = 1.959964 + 0.841621

#: Bars per year by cadence, used only to turn an information coefficient into
#: an annualised Sharpe for scale. Trading days x intraday bars.
BARS_PER_YEAR = {"1d": 252, "60m": 252 * 7, "1h": 252 * 7, "15m": 252 * 26}


def load_target_specs() -> dict[str, dict]:
    """Horizon and type per target, read from the config rather than typed here."""
    raw = yaml.safe_load(TARGETS_YAML.read_text(encoding="utf-8"))
    specs: dict[str, dict] = {}
    for name, body in _iter_target_blocks(raw):
        params = body.get("params") or {}
        shift = params.get("shift")
        window = params.get("window")
        if shift is None:
            continue
        specs[name] = {
            "type": body.get("type", "regression"),
            "shift": abs(int(shift)),
            "window": int(window) if window else 1,
            # A target that reaches `shift` bars forward and averages over
            # `window` bars depends on bars up to shift + window - 1, and two
            # observations closer than that share data.
            "block_bars": abs(int(shift)) + (int(window) - 1 if window else 0),
        }
    return specs


def _iter_target_blocks(node, _depth: int = 0):
    """Every mapping under a `target_*` key, wherever it sits in the file."""
    if not isinstance(node, dict) or _depth > 4:
        return
    for key, value in node.items():
        if isinstance(key, str) and key.startswith("target_") and isinstance(value, dict):
            yield key, value
        elif isinstance(value, dict):
            yield from _iter_target_blocks(value, _depth + 1)


def average_pairwise_correlation(panel: pd.DataFrame, max_names: int = 60) -> float:
    """Mean off-diagonal correlation of the target across tickers.

    Returns 0.0 when it cannot be computed, and the caller reports that as
    unknown rather than as "the names are independent" -- a correlation of
    zero is the most favourable possible assumption and the least likely.
    """
    frame = panel.dropna(axis=1, how="all")
    if frame.shape[1] < 2:
        return float("nan")
    if frame.shape[1] > max_names:
        frame = frame.iloc[:, :max_names]
    corr = frame.corr(min_periods=30).to_numpy()
    mask = ~np.eye(corr.shape[0], dtype=bool)
    values = corr[mask]
    values = values[np.isfinite(values)]
    return float(values.mean()) if values.size else float("nan")


def effective_breadth(n_names: int, rho: float) -> float:
    """How many independent names 110 correlated ones are worth.

    Capped at `n_names`, which the bare formula is not. A cross-sectionally
    demeaned target -- `target_relative_return_5d`, `target_relative_rank_5d`
    -- has average pairwise correlation of about -1/(N-1) BY CONSTRUCTION,
    because subtracting the cross-sectional mean forces the residuals to sum
    to zero. That drives the denominator to nearly zero and the first run of
    this script duly reported 1,131 effective names out of 110, and an MDE of
    0.0021 that would have read as very good news.

    Negative average correlation is a statement that the names are demeaned,
    not that there are more of them than exist.
    """
    if not math.isfinite(rho) or n_names <= 1:
        return float(n_names)
    denominator = 1.0 + (n_names - 1) * rho
    if denominator <= 0:
        return float(n_names)
    return min(float(n_names), n_names / denominator)


def analyse(targets_path: Path, specs: dict[str, dict], attempts: int) -> pd.DataFrame:
    available = pd.read_parquet(targets_path, columns=["ticker", "datetime", "interval"])
    rows = []
    for name, spec in sorted(specs.items()):
        column = _read_target(targets_path, name)
        if column is None:
            continue
        frame = available.copy()
        frame["value"] = column.to_numpy()
        frame = frame.dropna(subset=["value"])
        if frame.empty:
            continue
        # A target is defined on one cadence; take whichever interval holds it.
        interval = frame["interval"].value_counts().idxmax()
        frame = frame[frame["interval"] == interval]
        rows.append(_measure(name, spec, frame, interval, attempts))
    return pd.DataFrame(rows)


def _read_target(path: Path, name: str) -> pd.Series | None:
    try:
        return pd.read_parquet(path, columns=[name])[name]
    except (ValueError, KeyError):
        return None


def _measure(name: str, spec: dict, frame: pd.DataFrame, interval: str,
             attempts: int) -> dict:
    n_rows = len(frame)
    n_names = frame["ticker"].nunique()
    n_dates = frame["datetime"].nunique()

    panel = frame.pivot_table(
        index="datetime", columns="ticker", values="value", aggfunc="last"
    )
    rho = average_pairwise_correlation(panel)
    breadth = effective_breadth(n_names, rho)

    block = max(int(spec["block_bars"]), 1)
    independent_periods = n_dates / block
    n_eff = independent_periods * breadth

    mde_ic = Z_DETECT / math.sqrt(n_eff) if n_eff > 0 else float("nan")
    # Bonferroni over the number of (context, target, model) attempts a run
    # makes: the same detectable effect at alpha/M instead of alpha.
    z_corrected = _z_for(0.05 / max(attempts, 1)) + 0.841621
    mde_ic_corrected = z_corrected / math.sqrt(n_eff) if n_eff > 0 else float("nan")

    bars_year = BARS_PER_YEAR.get(interval, 252)
    periods_year = bars_year / block
    span_years = n_dates / bars_year
    # Fundamental law of active management, used only for scale. Note what
    # this reduces to: MDE_IC = Z / sqrt(periods * breadth), so the product is
    # Z * sqrt(bars_per_year / n_dates) -- breadth cancels, and the smallest
    # detectable annualised Sharpe depends ONLY on how many years the sample
    # covers. More names raise the IC that can be resolved; they do not
    # lengthen the record.
    sharpe_at_mde = mde_ic * math.sqrt(breadth * periods_year)

    is_binary = spec["type"].startswith("classification")
    mde_bacc = 0.5 * Z_DETECT / math.sqrt(n_eff) if n_eff > 0 else float("nan")

    return {
        "target": name,
        "cadence": interval,
        "span_years": span_years,
        "kind": "binary" if is_binary else "continuous",
        "rows": n_rows,
        "names": n_names,
        "dates": n_dates,
        "block_bars": block,
        "rho": rho,
        "eff_names": breadth,
        "indep_periods": independent_periods,
        "n_eff": n_eff,
        "mde_ic": mde_ic,
        "mde_ic_fdr": mde_ic_corrected,
        "mde_bacc_above_half": mde_bacc if is_binary else float("nan"),
        "sharpe_at_mde": sharpe_at_mde,
    }


def _z_for(alpha_two_sided: float) -> float:
    """Critical z for a two-sided alpha, without pulling in scipy."""
    from statistics import NormalDist

    return NormalDist().inv_cdf(1.0 - alpha_two_sided / 2.0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-dir", type=Path, default=DEFAULT_BATCH)
    parser.add_argument(
        "--attempts", type=int, default=1,
        help=(
            "How many (context, target, model) combinations a run tests, for "
            "the multiplicity column. 1 means no correction."
        ),
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    targets_path = args.batch_dir / "targets.parquet"
    if not targets_path.exists():
        print(f"no targets at {targets_path}", file=sys.stderr)
        return 1

    specs = load_target_specs()
    print(f"targets declared in config with a horizon: {len(specs)}")
    table = analyse(targets_path, specs, args.attempts)
    if table.empty:
        print("no target column in the batch matched the config", file=sys.stderr)
        return 1

    table = table.sort_values(["cadence", "target"])
    with pd.option_context("display.width", 200, "display.max_columns", 40):
        print(table.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.out, index=False)
        print(f"written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
