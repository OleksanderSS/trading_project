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

---

## Outcome — run 2026-08-20, nothing above edited

    yr    train   trades   selected    passive     excess
  1999    6,659    1,247   +0.01435   +0.00728   +0.00708
  2001   16,214    1,474   +0.00759   +0.00343   +0.00416
  2003   26,180    1,493   +0.01048   +0.00762   +0.00286

    folds positive: 3/3      mean excess +0.00470
    criterion:      >= 3/3   AND mean > +0.00050

### SUPPORTED

The pre-registered criterion is met, and this is the first result in this
project confirmed on data that took no part in choosing it.

### What the size means, and it is not what it looks like

The excess is **3.4x the exploratory estimate** (+0.00470 against +0.00138).
The document above warned that a confirmation reproducing the exploratory size
exactly would be suspicious; one this much larger is a different warning, and
it points at the limitation written down before the run rather than at a
stronger effect.

Measured after the fact, on the same data:

    cross-sectional dispersion (std of relative return)
      1999-2003 average   558 bp
      2015-2025 average   345 bp
      ratio               1.62x

Ranking has far more to work with when names move apart from each other, and
1999-2003 is the dot-com peak and bust. That accounts for much of the gap.

The rest is almost certainly the cost assumption, also named in advance: this
run charged 4.8 bp round trip, the modern IBKR figure. US equities were quoted
in SIXTEENTHS until 2001 — a minimum spread around 6 cents, or roughly 20 bp on
a $30 stock, four times what was charged. On the 1999 fold that alone would
consume a large part of +0.00708.

### Standing conclusion

The DIRECTION is confirmed out of sample: a cross-sectional regression target
beats passive holding on data never used to select it. The MAGNITUDE from this
window is not transferable to today's market and must not be used for sizing,
for projections, or in any conversation with an investor. The exploratory
figure of +0.0014 on 2005-2025 remains the honest estimate of the size, and it
is itself the sixth of six variants tried.

### What would make this stronger, in order

1. Re-run the confirmatory folds with a period-appropriate cost (20-30 bp
   before 2001, 10 bp to 2010, 4.8 bp after). If the direction survives that,
   it is not a cost artifact.
2. Repeat on tickers outside the 22 in this batch — an axis no variant has
   touched. Requires enriching them first.
3. Portfolio construction rather than per-trade means: positions overlap
   (buy daily, hold five), so the per-trade figure is not what an account
   would earn.

---

## Follow-up 1 — period-appropriate costs, and a correction to my own method

The section above said "the cost assumption flatters this period". **That was
wrong as written.** The confirmatory script computed

    excess = (selected_mean - COST) - (passive_mean - COST)

in which COST cancels exactly. Every excess figure reported that day was
insensitive to the cost assumption; it did nothing at all.

What that hid matters more than what it claimed. The strategy re-ranks and
re-enters every five days and pays a round trip each time; buy-and-hold pays
one round trip across the whole window. Charging both the same per-period cost
hands the strategy a subsidy equal to its own turnover.

Re-run with each side paying what it actually pays:

    cost      1999      2001      2003      2011      2019      2025   folds+
     0bp  +0.00708  +0.00416  +0.00286  +0.00051  +0.00207  +0.00519    6/6
     5bp  +0.00659  +0.00367  +0.00237  +0.00002  +0.00158  +0.00470    6/6
    10bp  +0.00610  +0.00318  +0.00188  -0.00047  +0.00109  +0.00421    5/6
    30bp  +0.00414  +0.00122  -0.00008  -0.00243  -0.00087  +0.00225    3/6

Break-even cost per fold, against what US equities actually cost then:

    1999    72.2 bp   vs ~20-30 bp (quoted in sixteenths)   survives
    2001    42.4 bp   vs ~20 bp                             survives
    2003    29.2 bp   vs ~10 bp                             survives
    2011     5.2 bp   vs ~5 bp                              MARGINAL
    2019    21.1 bp   vs ~5 bp                              survives
    2025    53.0 bp   vs ~5 bp                              survives

**The suspicion that 1999-2003 rode on an understated cost is not supported.**
Those folds break even at 72 and 42 bp against era spreads of 20-30. The
turnover asymmetry itself costs about 5 bp per period, small beside gross edges
of 200-700 bp — worth measuring rather than assuming in either direction.

## Follow-up 2 — these numbers CANNOT be annualised, and the reason is not minor

Every figure above is per five-day period, and multiplying by ~50 periods a
year gives 25%+, which would be false.

Positions overlap. Buying daily and holding five days means five concurrent
books, so roughly a fifth of capital stands behind any one of them. The "50
rebalances" in the break-even calculation counts re-entries, not independent
turns of the whole account.

**How much of a 14 bp per-period ranking edge reaches an account is still
unmeasured.** It is now the largest open question about this result: the
direction is confirmed twice, once on data that took no part in choosing it,
and the size has never been expressed as money an account would hold.

Nothing here may be converted to an annual return, quoted to an investor, or
used for sizing until the portfolio is actually constructed.
