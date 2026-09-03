"""The admission bar for ROADMAP 1.2, in the units the report already produces.

`leading_feature_report` answers "is this IC distinguishable from zero" and
answers it well: the IC is computed per DATE and the t is taken on the series
of daily coefficients, so pooled rows are not treated as independent, and
Benjamini-Hochberg corrects for the number of features tried.

What it does not answer is whether an IC that survives is WORTH ANYTHING. Those
are different questions, and after R17/R18 the second one has a number: the
pipeline refuses anything below an annualised Sharpe of about 1, so a feature
whose IC implies less than that will be refused downstream however significant
it is. Passing it on costs a promotion attempt, and attempts are what the
gate's bar is a correction for.

The fundamental law of active management gives the conversion:

    IR = IC * sqrt(BR),    BR = independent bets per year
                              = effective breadth * rebalances per year

Effective breadth is not the number of names. Correlated names are one bet
made repeatedly: N / (1 + (N-1) * rho_bar), which saturates at 1/rho_bar
however many names are added. That saturation is the whole reason 110 large
caps cannot become 1,100 by adding more of the same.
"""
import sys
from pathlib import Path

sys.path.insert(0, "D:/trading_project")

import numpy as np
import pandas as pd

BATCH = Path("D:/trading_project/data/colab/accumulated/main_database")

targets = pd.read_parquet(BATCH / "targets.parquet",
                          columns=["ticker", "datetime", "interval",
                                   "target_return_1d"])
daily = targets[targets["interval"] == "1d"].dropna(subset=["target_return_1d"])
wide = daily.pivot_table(index="datetime", columns="ticker",
                         values="target_return_1d")
# Only names with enough overlap to correlate honestly.
wide = wide.loc[:, wide.notna().sum() >= 500]
corr = wide.corr(min_periods=250)
values = corr.to_numpy()
mask = ~np.eye(len(values), dtype=bool)
rho = float(np.nanmean(values[mask]))
n = int(len(values))
breadth = n / (1 + (n - 1) * rho)

print(f"names with >= 500 daily returns : {n}")
print(f"mean pairwise correlation rho   : {rho:.4f}")
print(f"effective breadth N/(1+(N-1)rho): {breadth:.2f}")
print(f"saturation ceiling 1/rho        : {1/rho:.2f}   "
      f"(adding names cannot pass this)")

print(f"\n{'rebalance':<14}{'bets/year':>12}{'IC for IR 1.0':>16}"
      f"{'IC for IR 0.5':>16}")
print("-" * 58)
for label, per_year in (("daily", 252), ("weekly", 52), ("monthly", 12)):
    bets = breadth * per_year
    print(f"{label:<14}{bets:>12,.0f}{1.0/np.sqrt(bets):>16.4f}"
          f"{0.5/np.sqrt(bets):>16.4f}")

print("\nWhat the numbers mean for the report's own columns:")
print("  * ic_within is the column to hold to this, not ic_out: ic_out can be")
print("    carried by WHICH NAME rather than WHEN, and a book built on that is")
print("    one bet discovered after the fact (the tool's own docstring).")
print("  * a feature must clear BOTH: Benjamini-Hochberg on the date-series t")
print("    (already implemented) AND the IC above. The first says the effect")
print("    is there; the second says it is worth an attempt.")
print("  * BH controls the FALSE DISCOVERY rate, the gate controls the")
print("    FAMILY-WISE rate. Different guarantees; do not read one as the")
print("    other when the two reports are compared.")
