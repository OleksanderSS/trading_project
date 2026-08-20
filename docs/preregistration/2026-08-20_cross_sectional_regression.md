# Pre-registration: cross-sectional regression target

**Written and committed BEFORE the confirmatory run. Nothing below may be
edited after results are seen; a follow-up section records the outcome.**

## Why this document exists

Six variants were explored on 2026-08-20 in the walk-forward harness:

1. binary absolute target
2. binary relative, market-hedged
3. binary relative, unhedged
4. binary relative, next-open execution
5. regression on relative return, hyperparameters chosen by MSE
6. regression on relative return, hyperparameters chosen by rank IC

The last two scored best — 10 of 11 folds positive, excess over passive
+0.00138 and +0.00167, t 2.81 and 4.13. Reporting that as a finding would
repeat, one level up, the exact defect this audit has spent a week removing:
the threshold used to be chosen on the rows it was scored on, and now the
VARIANT would be chosen on the folds it is scored on. A t-statistic that does
not account for six attempts is not a t-statistic.

Every fold with a test year from 2005 to 2025 has been seen. The years
**1996–2004 have never been a test fold** — they entered only as training
data — so they are the one segment of this dataset untouched by the choice of
variant.

## Hypothesis

Predicting the CROSS-SECTIONAL relative return as a regression target, then
buying the top 30% by predicted value at each date, produces a positive excess
over holding everything equally, on data not used to select this variant.

## Procedure, fixed in advance

- Data: `data/colab/accumulated/main_database/features.parquet`, interval `1d`.
- Payoff: `open[t+6] / open[t+1] - 1`. Entry at the next open, never at the
  close that generated the signal.
- Target: that payoff minus its cross-sectional mean at the same timestamp.
- Features: top 35 by |Pearson correlation| with the target, computed on the
  training rows of the fold ONLY.
- Model: `LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=31,
  min_child_samples=20, subsample=0.8, colsample_bytree=0.8, random_state=0)`.
  Fixed, not searched — a search would reintroduce a selection step.
- Folds: expanding training window from 1996, test years **1999, 2001, 2003**.
  No fold overlaps another's test window.
- Decision: buy the top 30% by predicted value within each date.
- Cost: 4.8 bp round trip, the shipped `ibkr_pro_tiered` profile.
- Metric: mean payoff of the selected rows minus the mean payoff of all rows in
  the same test window, both after cost. Excess over passive holding.

## Acceptance criterion, fixed in advance

The hypothesis is SUPPORTED only if **excess > 0 in at least 3 of the 3 folds**
and the mean excess exceeds **+0.0005** — roughly half the +0.0014 seen in the
exploratory runs, since a first out-of-sample confirmation that reproduces the
exploratory size exactly would itself be suspicious.

Anything less is recorded as NOT SUPPORTED. Three folds cannot establish an
effect; they can only fail to contradict it, and that limit is stated here so
the result is not overread afterwards.

## Known limitations, stated before seeing the result

- Three folds is a small sample. A positive outcome is weak evidence FOR;
  a negative outcome is strong evidence AGAINST, because the exploratory
  estimate was large.
- 1999–2003 spans the dot-com peak and bust. Market structure differed:
  decimalisation arrived in 2001, and spreads were wider than the 4.8 bp
  charged here. The cost assumption flatters this period.
- Training windows are shorter than in the exploratory runs (3–7 years rather
  than 8–29), which works against the hypothesis.

## Outcome

*To be completed after the run, without editing anything above.*
