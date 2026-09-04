"""Columns capable of carrying a signal must not accumulate unmeasured.

ROADMAP 1.2's third bullet -- leadingness as an ADMISSION criterion rather
than a post-hoc report -- turned out to be three gates, not one, because
`features.yaml` enables 22 ENRICHERS and not features: there is no lever that
says "this column enters because it was measured" (CLAIMS R29).

    A  enricher level: turn off enrichers whose whole output is incapable
    B  book construction: a degenerate column must not become a position
    C  this file: the first two must not rot

C exists because A and B have a known failure mode in this repository, with
names and dates. `critical: true` sat in collectors.yaml for twenty
collectors and was read NOWHERE in src/ (#252). `family_size` was promised in
one config and verified in none (#235). Both were mechanisms that existed on
paper, and both were found by accident months later. A gate with no ratchet
becomes a comment.

WHAT IS RATCHETED, and why these two numbers rather than the obvious one.

    capable but never measured   4    columns that CAN carry a cross-sectional
                                      signal and for which no leadingness
                                      verdict exists. This is the admission
                                      criterion itself, counted.

    tie hazard                  15    columns whose commonest value covers
                                      >= 90% of the panel. These rank as ties,
                                      `sign(rank - 0.5)` sends every name the
                                      same way, and the book is the market.
                                      Seven such columns cleared a Bonferroni
                                      correction at net Sharpe 1.016 on
                                      2026-09-04 while the constant opponent
                                      scored 1.018 (CLAIMS R28).

NOT ratcheted: the count of incapable columns (1,166 of 1,390). Two thirds of
the batch is intraday data sitting entirely inside the sealed period, and
intraday collection continues by the owner's decision. A ceiling there would
fail every run for a reason nobody intends to fix, and a check that fires on
ordinary conditions gets switched off -- `|| true` sat in ci.yml for six weeks
for exactly that.

WHY THIS READS COMMITTED CSVs AND NOT THE BATCH. `features.parquet` is 888 MB
and is not in the repository, so a ratchet that needed it would skip in CI --
and a skip is not a pass. Two financial-math tests skipped from the day they
were written on module paths that did not exist, and the suite reported green
for as long as they existed (#261). So the counts come from the committed
reports, and the one check that genuinely needs the batch is separated out,
skipping with a reason that states exactly what went unverified.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INVENTORY = PROJECT_ROOT / "diagnostic_reports" / "batch_inventory_1d.csv"
ROLES = PROJECT_ROOT / "diagnostic_reports" / "feature_roles_1d.csv"
TOOL = PROJECT_ROOT / "scripts" / "diagnostics" / "what_has_never_been_measured.py"
BATCH = (PROJECT_ROOT / "data" / "colab" / "accumulated" / "main_database"
         / "features.parquet")

#: Counted 2026-09-04 on the first full inventory. Ceilings may only FALL.
#:
#: Raising one would hide exactly what it was written to show, and a ceiling
#: raised once is raised again -- the reason `_silent_failure_scan` refuses it
#: in its own words.
CAPABLE_BUT_UNMEASURED_CEILING = 4
TIE_HAZARD_CEILING = 15

#: The verdict strings the tool writes. Restated here deliberately: if the
#: tool's wording changes, this file must be re-read rather than silently
#: matching nothing and passing. A ratchet that counts zero because its filter
#: stopped matching is the most comfortable kind of broken.
CAPABLE = "can carry a cross-sectional signal"
TIE_HAZARD_PREFIX = "tie hazard"


def capable_but_unmeasured(frame: pd.DataFrame) -> pd.DataFrame:
    """Columns able to carry a cross-sectional signal with no measured role.

    A function rather than two lines inside a test, so that
    `test_the_counters_catch_their_own_case` can prove it still counts. A
    counter that quietly stopped matching would pass every ceiling in this
    file and report a clean batch.
    """
    capable = frame[frame["verdict"] == CAPABLE]
    return capable[~capable["measured"].astype(bool)]


def tie_hazards(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[frame["verdict"].astype(str).str.startswith(TIE_HAZARD_PREFIX)]


@pytest.fixture(scope="module")
def inventory() -> pd.DataFrame:
    assert INVENTORY.exists(), (
        f"{INVENTORY.relative_to(PROJECT_ROOT)} is missing. It is committed on "
        f"purpose: without it there is nothing to ratchet, and the gate becomes "
        f"a plan again. Rebuild with `python {TOOL.relative_to(PROJECT_ROOT)}`."
    )
    return pd.read_csv(INVENTORY)


def test_the_verdicts_this_file_counts_still_exist(inventory):
    """A filter that matches nothing passes every ceiling."""
    verdicts = set(inventory["verdict"])
    assert CAPABLE in verdicts, (
        f"no column carries the verdict {CAPABLE!r}; either the tool's wording "
        f"changed or the report is empty, and every count below is meaningless"
    )
    assert any(v.startswith(TIE_HAZARD_PREFIX) for v in verdicts), (
        f"no column carries a {TIE_HAZARD_PREFIX!r} verdict. If the shape "
        f"genuinely disappeared, lower TIE_HAZARD_CEILING to 0 deliberately "
        f"rather than leaving a filter that matches nothing"
    )


def test_capable_columns_do_not_accumulate_unmeasured(inventory):
    """The admission criterion, counted.

    A column able to carry a cross-sectional signal and never measured for
    leadingness is one that entered the batch because somebody wrote it. That
    is the thing 1.2 exists to stop.
    """
    unmeasured = capable_but_unmeasured(inventory)

    assert len(unmeasured) <= CAPABLE_BUT_UNMEASURED_CEILING, (
        f"capable-but-unmeasured columns rose from "
        f"{CAPABLE_BUT_UNMEASURED_CEILING} to {len(unmeasured)}. Each one is a "
        f"feature admitted without a leadingness verdict:\n"
        + "\n".join(f"  {name}" for name in unmeasured["feature"].head(20))
    )


def test_the_tie_hazard_does_not_spread(inventory):
    """The R28 shape: 98.9% one value, ranks are ties, book is the market."""
    hazard = tie_hazards(inventory)

    assert len(hazard) <= TIE_HAZARD_CEILING, (
        f"tie-hazard columns rose from {TIE_HAZARD_CEILING} to {len(hazard)}. "
        f"A column whose commonest value covers most of the panel cannot "
        f"produce a two-sided book:\n"
        + "\n".join(
            f"  {row.feature}  mode_share={row.mode_share:.3f}"
            for row in hazard.nlargest(20, "mode_share").itertuples())
    )


def test_the_two_reports_describe_the_same_batch(inventory):
    """The inventory says what CAN carry a signal; the roles report says what
    DOES. If one names a column the other has never heard of, they were built
    from different batches and neither count means anything."""
    if not ROLES.exists():
        pytest.fail(
            f"{ROLES.relative_to(PROJECT_ROOT)} is missing, so `measured` in "
            f"the inventory cannot be checked against anything"
        )
    roles = set(pd.read_csv(ROLES)["feature"])
    catalogued = set(inventory["feature"])
    orphans = sorted(roles - catalogued)

    assert not orphans, (
        f"{len(orphans)} features have a measured role but no inventory entry, "
        f"so the two reports were built from different batches:\n"
        + "\n".join(f"  {name}" for name in orphans[:20])
    )


def test_measured_matches_the_roles_report(inventory):
    """`measured` is a derived column and derived columns drift. Recomputing it
    costs nothing and catches a report edited by hand."""
    roles = set(pd.read_csv(ROLES)["feature"])
    claimed = set(inventory[inventory["measured"].astype(bool)]["feature"])
    assert claimed == (roles & set(inventory["feature"])), (
        "the inventory's `measured` column disagrees with feature_roles_1d.csv"
    )


def test_the_tool_that_builds_this_still_declares_its_thresholds():
    """The ceilings above are only meaningful under the rules the report was
    built with. If TIE_HAZARD moves from 0.90, the count changes for a reason
    that has nothing to do with the batch."""
    assert TOOL.exists(), f"{TOOL.relative_to(PROJECT_ROOT)} is gone"
    source = TOOL.read_text(encoding="utf-8")
    assert "TIE_HAZARD = 0.90" in source, (
        "the tie-hazard threshold moved. The ceiling in this file counts "
        "columns under the old rule; re-run the inventory and re-derive it "
        "deliberately."
    )
    assert "MIN_HISTORY = 10_000" in source, (
        "the history threshold moved; the capable/incapable split changed "
        "underneath these ceilings"
    )


@pytest.mark.skipif(
    not BATCH.exists(),
    reason=(
        "features.parquet is absent, so the inventory cannot be checked "
        "against the batch it describes. UNVERIFIED WHILE SKIPPED: whether "
        "columns were added to the batch without re-running "
        "what_has_never_been_measured.py, which would leave every ceiling in "
        "this file counting a stale report. The ceilings themselves still ran."
    ),
)
def test_the_inventory_is_not_stale_against_the_batch(inventory):
    """The one check that needs the data. A ratchet over a snapshot passes
    happily while the thing it describes moves on."""
    import pyarrow.parquet as pq

    ident = {"ticker", "datetime", "interval"}
    columns = {
        field.name for field in pq.ParquetFile(BATCH).schema_arrow
        if field.name not in ident
    }
    catalogued = set(inventory["feature"])
    missing = sorted(columns - catalogued)

    assert not missing, (
        f"{len(missing)} columns are in the batch and absent from the "
        f"inventory, so the report is stale and every ceiling here is counting "
        f"a batch that no longer exists. Re-run "
        f"`python {TOOL.relative_to(PROJECT_ROOT)}`:\n"
        + "\n".join(f"  {name}" for name in missing[:20])
    )


def test_the_counters_catch_their_own_case():
    """A ratchet that cannot fail its own case proves nothing.

    This is the shape of the two rows that would matter: a capable column
    nobody measured, and the column that scored net Sharpe 1.016 on
    2026-09-04 by being 98.9% one value. Both are constructed here rather
    than taken from the report, so the counters are tested against something
    the report cannot quietly stop containing.
    """
    planted = pd.DataFrame([
        {"feature": "some_new_column_1d", "measured": False,
         "verdict": CAPABLE, "mode_share": 0.12},
        {"feature": "already_checked_1d", "measured": True,
         "verdict": CAPABLE, "mode_share": 0.31},
        {"feature": "state_CDL_HAMMER_1d", "measured": True,
         "verdict": "tie hazard: ranks are ties, book would be one-sided",
         "mode_share": 0.989},
        {"feature": "an_intraday_column_60m", "measured": False,
         "verdict": "too little history", "mode_share": 1.0},
    ])

    unmeasured = capable_but_unmeasured(planted)
    assert list(unmeasured["feature"]) == ["some_new_column_1d"], (
        "the capable-but-unmeasured counter no longer recognises a capable "
        "column without a measured role"
    )

    hazard = tie_hazards(planted)
    assert list(hazard["feature"]) == ["state_CDL_HAMMER_1d"], (
        "the tie-hazard counter no longer recognises the shape it was written "
        "for"
    )

    # And it must not count what it is not for: an incapable column is dead
    # weight, not an admission failure, and counting it here would make the
    # ceiling fire on two thirds of the batch every run.
    assert "an_intraday_column_60m" not in set(unmeasured["feature"])
    assert "an_intraday_column_60m" not in set(hazard["feature"])
