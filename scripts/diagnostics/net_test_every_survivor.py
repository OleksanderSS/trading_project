"""The net test, applied to every feature that survived 1.2 -- not just the winner.

R22 ran the holding sweep on ONE feature, the single one whose verdict was
"survives, worth testing", and found it untradeable at every turnover. That
closes the question for that feature and not for the report.

Forty-six features passed Benjamini-Hochberg at 5% FDR. Their verdicts were
"survives but tiny", "faded", "labels the name" -- all reasons to doubt, none
of them "cannot be traded", because until R22 nobody had asked about cost. A
feature with a weaker IC but a slower decay would beat the winner at a longer
hold, and the ranking in 1.2 cannot see that: it ranks by strength at one bar,
not by what survives friction.

So the question this answers is not "is there a better feature" but "does the
conclusion depend on which feature we picked". If all 46 are negative net at
every horizon, the finding is about the COST STRUCTURE and the universe, and
looking for more features in the same batch is looking in the wrong place.

Both legs pay the friction, the Sharpe is the portfolio's, and holding periods
are sampled without overlap. The sealed period is untouched.

    python scripts/diagnostics/net_test_every_survivor.py
    python scripts/diagnostics/net_test_every_survivor.py --holds 1 20 120
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from src.targets.calculators.regression_calculator import (  # noqa: E402
    RegressionCalculator,
)

BATCH = PROJECT_ROOT / "data" / "colab" / "accumulated" / "main_database"
ROLES = PROJECT_ROOT / "diagnostic_reports" / "feature_roles_1d.csv"
SEALED = pd.Timestamp("2023-09-01", tz="UTC")

#: The standard error of an annualised Sharpe over T years is about
#: sqrt(1/T); over the 27 explorable years that is 0.193.
#:
#: TWO of those is NOT enough here, and the first version of this script said
#: it was. This sweep makes 46 x 5 = 230 attempts and reports the maximum, and
#: `best_hold` inside each feature is already a max over five. The expected
#: maximum of 230 draws from pure noise is 2.63 standard errors -- 0.507 -- and
#: family-wise 5% by Bonferroni needs 3.70, which is 0.714. A threshold chosen
#: without counting the attempts is exactly the defect the promotion gate spent
#: a week having removed (CLAIMS R11, R17), reproduced in the script that was
#: meant to judge its output.
SHARPE_SE = 0.193
N_ATTEMPTS = 46 * 5
NOISE_MAX = 0.507      # expected max of N_ATTEMPTS draws, in Sharpe
BONFERRONI = 0.714     # family-wise 5% over N_ATTEMPTS, in Sharpe


def _sharpe(series: np.ndarray, per_year: float) -> float:
    usable = series[np.isfinite(series)]
    if usable.size < 30 or usable.std() <= 0:
        return float("nan")
    return float(usable.mean() / usable.std() * np.sqrt(per_year))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holds", type=int, nargs="+", default=[1, 5, 20, 40, 120])
    args = parser.parse_args()

    roles = pd.read_csv(ROLES)
    survivors = roles[(roles["passes_fdr"]) & (roles["varies"] > 0.5)]
    names = [n for n in survivors["feature"].tolist()]
    print(f"{len(names)} features passed FDR with real cross-sectional variation\n")

    costs = yaml.safe_load(
        (PROJECT_ROOT / "src/config/targets.yaml").read_text(encoding="utf-8")
    )["targets"]["target_return_1d"]["params"]["transaction_costs"]

    ident = ["ticker", "datetime", "interval"]
    # Deduplicated: `close` is itself one of the 46 survivors, and asking
    # parquet for it twice returns a DataFrame under that name rather than a
    # Series, which the cost model then refuses.
    wanted = list(dict.fromkeys(ident + ["close"] + names))
    frame = pd.read_parquet(BATCH / "features.parquet", columns=wanted)
    frame = frame[frame["interval"] == "1d"].copy()
    frame["datetime"] = pd.to_datetime(frame["datetime"], utc=True)
    frame = frame[frame["datetime"] < SEALED]
    frame = frame.dropna(subset=["close"]).sort_values(["ticker", "datetime"])
    frame = frame.reset_index(drop=True)
    friction = np.asarray(
        RegressionCalculator._round_trip_cost(frame["close"], costs), dtype=float)

    forwards = {}
    for hold in args.holds:
        forwards[hold] = (frame.groupby("ticker", sort=False)["close"]
                          .transform(lambda s: s.shift(-hold) / s - 1.0)
                          .to_numpy())

    header = f"{'feature':<34}" + "".join(f"{'h' + str(h):>9}" for h in args.holds) + \
             f"{'best net':>10}{'at hold':>9}"
    print(header)
    print("-" * len(header))

    rows = []
    for name in names:
        values = pd.to_numeric(frame[name], errors="coerce")
        if values.notna().sum() < 10_000:
            continue
        position = np.sign(
            values.groupby(frame["datetime"]).rank(pct=True).to_numpy() - 0.5)
        position = np.nan_to_num(position)
        work = pd.DataFrame({"datetime": frame["datetime"].to_numpy()})
        nets = {}
        for hold in args.holds:
            net = position * forwards[hold] - np.abs(position) * friction
            work["net"] = net
            by_date = work.groupby("datetime")["net"].mean().sort_index()
            nets[hold] = _sharpe(by_date.to_numpy()[::hold], 252.0 / hold)
        best_hold = max(nets, key=lambda h: (nets[h] if np.isfinite(nets[h]) else -9))
        rows.append({"feature": name, "best_net": nets[best_hold],
                     "best_hold": best_hold, **{f"h{h}": nets[h] for h in args.holds}})
        print(f"{name:<34}" + "".join(f"{nets[h]:>9.3f}" for h in args.holds) +
              f"{nets[best_hold]:>10.3f}{best_hold:>9}", flush=True)

    report = pd.DataFrame(rows)
    if report.empty:
        print("\nnothing measurable")
        return 1

    print("\n" + "=" * len(header))
    real = report[report["best_net"] >= BONFERRONI]
    weak = report[(report["best_net"] > 0) & (report["best_net"] < BONFERRONI)]
    print(f"features tested                              {len(report)}")
    print(f"best net Sharpe anywhere                     {report['best_net'].max():+.3f}")
    print(f"clear Bonferroni for {N_ATTEMPTS} attempts ({BONFERRONI:.3f})  {len(real)}")
    print(f"clear the expected noise maximum ({NOISE_MAX:.3f})       "
          f"{int((report[chr(39)+chr(39)] if False else report['best_net'] >= NOISE_MAX).sum())}")
    print(f"positive but inside the noise                {len(weak)}")
    print(f"negative at every horizon                    "
          f"{int((report['best_net'] <= 0).sum())}")
    if len(real):
        print("\nthese clear the noise and are the first real candidates:")
        print(real[["feature", "best_net", "best_hold"]].to_string(index=False))
    else:
        print("\nNONE clears two standard errors. The conclusion is not about "
              "which feature was picked:\nit is about the cost structure and "
              "the universe, and more features from this batch\nwill be found "
              "in the same place.")
    out = PROJECT_ROOT / "diagnostic_reports" / "net_test_survivors.csv"
    report.to_csv(out, index=False)
    print(f"\nwritten to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
