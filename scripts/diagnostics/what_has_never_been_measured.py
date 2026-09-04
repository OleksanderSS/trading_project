"""How much of the batch has never been measured, and what is structurally dead.

ROADMAP 1.2's third bullet says leadingness must become an ADMISSION criterion
rather than a post-hoc report: a feature should enter the batch because it was
measured to carry information, not because someone wrote it. Nothing can be
gated that has not been inventoried, and the inventory did not exist.

The counts that motivate this: `features.parquet` holds **1,390 feature
columns**. `feature_roles_1d.csv` carries a measured verdict for **455**. So
**935 columns have never been measured for anything at all** -- not for
leadingness, not for variation, not for whether they are constant.

WHY THAT IS A HAZARD AND NOT MERELY UNTIDY. On 2026-09-04 a column that is
98.9% one value produced a "long/short book" that was long every name, scored
a net Sharpe of 1.016, and cleared a Bonferroni correction -- because it was
the market wearing a feature's name (CLAIMS R28). That column was one of the
455 that HAD been measured. What sits in the other 935 is unknown, and the
same shape there is invisible.

WHAT THE MULTIPLICITY ARGUMENT IS ACTUALLY WORTH -- stated because it is the
argument everyone reaches for first, and it is the weaker one. The honest
threshold grows as sqrt(log N), so it is nearly flat:

    family        Bonferroni    noise max
    46 x 6            0.723        0.525
    235 x 6           0.798        0.621
    1390 x 6          0.874        0.713

Cutting the batch from 1,390 features to 46 lowers the bar by 17%. Real, but
not the reason to build a gate. The reasons that survive measurement are:
a column nobody measured cannot be known to be dead; a degenerate column can
score like a signal; and every unmeasured column is paid for in every rebuild.

WHAT IS MEASURED HERE, and each column exists because something can hide in it:

    history       rows with a value. Attention was collected 30 days deep
                  against a frame spanning decades: the column existed,
                  reported success, and could never be trained on.

    varies        Does the column differ BETWEEN names on the same date? A
                  market-wide series contributes exactly zero to a ranking
                  however good its correlation looks.

    mode_share    The largest single value's share of the rows. This is the
                  R28 hazard, and it is the one nobody was measuring: above
                  ~0.9 the ranks are almost all ties, `sign(rank - 0.5)` gives
                  the same side to every name, and the "book" is the market.

    uniq          Distinct values. Two, with a mode share near 1, is a flag
                  pretending to be a feature.

Nothing here says a feature is good. It says whether a feature is capable of
carrying a cross-sectional signal at all -- which is the cheapest of the
admission conditions and the only one that costs no target.

    python scripts/diagnostics/what_has_never_been_measured.py
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

BATCH = PROJECT_ROOT / "data" / "colab" / "accumulated" / "main_database"
ROLES = PROJECT_ROOT / "diagnostic_reports" / "feature_roles_1d.csv"
OUT = PROJECT_ROOT / "diagnostic_reports" / "batch_inventory_1d.csv"
SEALED = pd.Timestamp("2023-09-01", tz="UTC")
IDENT = {"ticker", "datetime", "interval"}
CHUNK = 40

#: A column whose commonest value covers this much of the panel cannot produce
#: a two-sided book: the ranks are ties, and `sign(rank - 0.5)` sends every
#: name the same way. Set at the level that would have caught R28's seven --
#: state_CDL_HAMMER_1d sits at 0.989 -- while leaving genuinely skewed but
#: usable columns alone.
TIE_HAZARD = 0.90

#: Below this many rows a column cannot be trained on across 27 years of daily
#: bars, whatever its correlation looks like on the rows it has.
MIN_HISTORY = 10_000


def main() -> int:
    schema = pq.ParquetFile(BATCH / "features.parquet").schema_arrow
    features = [f.name for f in schema if f.name not in IDENT]
    measured = set(pd.read_csv(ROLES)["feature"]) if ROLES.exists() else set()

    ident = pd.read_parquet(BATCH / "features.parquet",
                            columns=["ticker", "datetime", "interval"])
    ident["datetime"] = pd.to_datetime(ident["datetime"], utc=True)
    keep = (ident["interval"] == "1d") & (ident["datetime"] < SEALED)
    order = ident.index[keep].to_numpy()
    dates = ident.loc[keep, "datetime"].to_numpy()
    panel_rows = int(keep.sum())

    print(f"feature columns in the batch      {len(features):,}")
    print(f"carrying a measured role          {len(measured & set(features)):,}")
    print(f"NEVER measured for anything       "
          f"{len(set(features) - measured):,}")
    print(f"daily pre-sealed panel            {panel_rows:,} rows, "
          f"{len(np.unique(dates)):,} dates\n")

    rows = []
    for start in range(0, len(features), CHUNK):
        block = features[start:start + CHUNK]
        raw = (pd.read_parquet(BATCH / "features.parquet", columns=block)
               .iloc[order])
        # DO NOT COERCE BLINDLY. The first version ran `to_numeric(errors=
        # "coerce")` over every column, which turns a string column into all
        # NaN and reports it as "too little history". Nine columns are strings
        # here -- context fingerprints and pattern sequences -- and two of them
        # are fully populated on the daily panel. A tool built to find columns
        # nobody measured must not manufacture empty ones.
        numeric = [c for c in block
                   if pd.api.types.is_numeric_dtype(raw[c])
                   or pd.api.types.is_bool_dtype(raw[c])]
        other = [c for c in block if c not in numeric]

        history = raw.notna().sum()
        uniq = raw.nunique()
        mode = raw.apply(
            lambda s: float(s.value_counts(normalize=True).iloc[0])
            if s.notna().any() else 1.0)

        varies = {}
        if numeric:
            grouped = raw[numeric].groupby(dates)
            highest, lowest = grouped.max(), grouped.min()
            varies.update(((highest != lowest) & highest.notna()).mean().to_dict())
        for name in other:
            # nunique per date works on any dtype and is only paid for the
            # handful of columns that need it.
            varies[name] = float(
                (raw[name].groupby(dates).nunique() > 1).mean())

        for name in block:
            rows.append({
                "feature": name,
                "measured": name in measured,
                "dtype": str(raw[name].dtype),
                "history": int(history[name]),
                "coverage": float(history[name]) / panel_rows,
                "uniq": int(uniq[name]),
                "mode_share": float(mode[name]),
                "varies": float(varies[name]),
            })
        del raw
        print(f"  ...{start + len(block):>5} of {len(features)}", flush=True)

    report = pd.DataFrame(rows)

    # The verdict is about CAPACITY, not quality: can this column carry a
    # cross-sectional signal at all? Ordered so the first true reason wins,
    # because a column with no history has no meaningful mode share.
    def verdict(row) -> str:
        if row["history"] < MIN_HISTORY:
            return "too little history"
        if row["uniq"] <= 1:
            return "constant"
        if row["varies"] < 0.5:
            return "market-wide: no cross-sectional variation"
        if row["mode_share"] >= TIE_HAZARD:
            return "tie hazard: ranks are ties, book would be one-sided"
        return "can carry a cross-sectional signal"

    report["verdict"] = report.apply(verdict, axis=1)
    report = report.sort_values(["verdict", "feature"]).reset_index(drop=True)

    print("\n" + "=" * 74)
    print("CAPACITY, over the whole batch")
    print("=" * 74)
    counts = report["verdict"].value_counts()
    for name, count in counts.items():
        print(f"  {name:<52}{count:>6}  ({count / len(report):>5.1%})")

    print("\nsplit by whether anyone has ever measured the column")
    table = pd.crosstab(report["verdict"], report["measured"])
    table.columns = ["never measured", "measured"][:len(table.columns)]
    print(table.to_string())

    hazard = report[report["verdict"].str.startswith("tie hazard")]
    print(f"\nTIE HAZARD -- the R28 shape, {len(hazard)} columns. These rank as "
          f"ties, so a\n`sign(rank - 0.5)` book sends every name the same way "
          f"and scores like the market.")
    if len(hazard):
        worst = hazard.nlargest(10, "mode_share")
        print(worst[["feature", "uniq", "mode_share", "measured"]]
              .to_string(index=False))
        unmeasured = int((~hazard["measured"]).sum())
        print(f"\n  of those, {unmeasured} have never been measured for "
              f"anything -- the same\n  shape that scored 1.016 on 2026-09-04, "
              f"sitting where nobody has looked.")

    usable = report[report["verdict"] == "can carry a cross-sectional signal"]
    print(f"\nWHAT AN ADMISSION GATE WOULD ACTUALLY ADMIT, on capacity alone:")
    print(f"  columns able to carry a cross-sectional signal   {len(usable)}")
    print(f"  of which never measured for leadingness          "
          f"{int((~usable['measured']).sum())}")
    print(f"  columns that cannot, and are computed anyway     "
          f"{len(report) - len(usable)}")

    report.to_csv(OUT, index=False)
    print(f"\nwritten to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
