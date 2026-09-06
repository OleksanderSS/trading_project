"""A risk setting that no code reads is a promise the config cannot keep.

`risk_management.yaml` reads, to a human, as a description of a risk system:

    trading.risk_controls.pre_trade_var_check: true
    trading.risk_controls.position_size_limits: true
    liquidity_risk.min_volume_threshold: 1000000

Measured 2026-09-06: those three, and two more, appear NOWHERE in any Python
file in this repository -- not in `src/`, not in tests, not even in the archive.
A boolean set to `true` beside the words "pre trade var check" states that the
check happens. It does not happen. That is worse than a missing feature: it is a
safety guarantee that reads as configured.

WHAT THIS TEST IS AND IS NOT. It catches "declared and never mentioned". It
does NOT catch "mentioned and never used" -- `VirtualPortfolio` reads four
limits into variables and no line consults them afterwards (REGISTER #101), and
a name-grep sees those as read. Dead-store analysis is a linter's job and is not
attempted here; #101 stays open on its own terms.

THE MIRROR DIRECTION, MEASURED BY HAND AND DELIBERATELY NOT AUTOMATED. The
sharper defect is the other way round: code asking the risk block for a key the
config never declares, so `.get(key, default)` silently returns the number
written in the code and the config a person reads is not the config in force.
Measured 2026-09-06, six such keys -- `max_position_size` and `max_total_risk`
(`VirtualPortfolio`, defaults 0.1 and 0.3), `risk_per_trade_pct`
(`PortfolioManager`, 0.03), `correlation_threshold`, `max_asset_weight` and
`risk_limits` (`MaxExposureMonitor`, 0.7 / 0.25 / {}). The config declares
`max_position_size_pct`, a DIFFERENT name, so its value has never once reached
`VirtualPortfolio`; both happen to be 0.1, which is why nobody noticed.

Two attempts to make that a test both failed, in opposite directions. Following
imports out of every file naming the block MISSED `max_exposure_monitor.py`,
which receives it as a plain `config=` argument, and PULLED IN
`pipeline_policy_manager.py`, which reads a different block. Scoping by package
instead reported thirty-two keys, nearly all of them belonging to other configs
entirely -- `rsi_weight`, `sentiment_col`, `hrp_linkage`. There is no reliable
static way to tell which `config.get(key)` reads THIS block, and a blocking
test whose scope cannot be stated honestly is worse than none. So the finding
lives in REGISTER #288 as a measurement, and this file guards only the
direction it can guard truthfully.

WHY A LEDGER AND NOT A ZERO. Nine settings are in this state today, and
implementing them is not wiring -- the NUMBERS are a risk policy the owner has
to choose, and trading sits behind the pipeline for now (#101). A test that
demanded zero would be red today and stay red, which is how gates get switched
off. So the existing ones are written down by name, and the test blocks the
TENTH. The ledger may only shrink: a setting that acquires a reader is reported
as a gain to be banked in the same commit.

This is the ratchet this project already uses six times, most recently for the
unit suite (`tests/known_unit_failures.txt`).
"""
from __future__ import annotations

from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG = PROJECT_ROOT / "src" / "config" / "risk_management.yaml"
LEDGER = PROJECT_ROOT / "tests" / "known_unread_risk_settings.txt"

#: Leaf keys that are prose, not settings. A description has nothing to read it
#: by design, and listing every one in the ledger would bury the real entries.
PROSE_KEYS = {"description", "descriptions", "name", "label", "comment"}


def _leaf_settings(node, path=()):
    """Every leaf that carries a NUMBER or a BOOLEAN, with its full path."""
    if isinstance(node, dict):
        for key, value in node.items():
            yield from _leaf_settings(value, path + (str(key),))
    elif isinstance(node, (int, float, bool)) and not isinstance(node, str):
        if path and path[-1] not in PROSE_KEYS:
            yield ".".join(path), path[-1]


def _python_sources():
    for path in (PROJECT_ROOT / "src").rglob("*.py"):
        if "archive" in path.parts or "__pycache__" in path.parts:
            continue
        yield path


def _unread_settings():
    settings = dict(_leaf_settings(yaml.safe_load(CONFIG.read_text(encoding="utf-8"))))
    blob = "\n".join(p.read_text(encoding="utf-8", errors="ignore")
                     for p in _python_sources())
    return {full for full, leaf in settings.items() if leaf not in blob}


def _ledger() -> set[str]:
    path = LEDGER
    if not path.exists():
        return set()
    return {line.split("#")[0].strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")} - {""}


def test_no_new_risk_setting_is_declared_without_a_reader():
    unread = _unread_settings()
    ledger = _ledger()
    fresh = sorted(unread - ledger)
    assert not fresh, (
        "these risk settings are declared in risk_management.yaml and read by "
        "no code in src/:\n    " + "\n    ".join(fresh) + "\n\n"
        "A setting nothing reads is a promise the config cannot keep, and a "
        "boolean set to true\nreads as a guarantee that the check happens. "
        "Either give it a reader, or delete it,\nor -- if it is deliberately "
        "waiting on the owner's risk policy (#101) -- add it to\n"
        f"{LEDGER.relative_to(PROJECT_ROOT)} with the reason on the line.")


def test_the_ledger_does_not_keep_settings_that_now_have_readers():
    """A ceiling that never falls is a ceiling nobody is paying down."""
    stale = sorted(_ledger() - _unread_settings())
    assert not stale, (
        "these are in the ledger but now HAVE a reader:\n    "
        + "\n    ".join(stale)
        + "\n\nThat is a gain. Remove them from "
        f"{LEDGER.relative_to(PROJECT_ROOT)} in the same commit that earned it, "
        "so the\nceiling falls and cannot be climbed back.")


def test_every_ledger_line_says_why():
    """A bare name in a ledger becomes permanent; a reason can be argued with."""
    if not LEDGER.exists():
        return
    naked = [line.strip() for line in LEDGER.read_text(encoding="utf-8").splitlines()
             if line.strip() and not line.lstrip().startswith("#") and "#" not in line]
    assert not naked, (
        "ledger lines without a reason after '#':\n    " + "\n    ".join(naked)
        + "\n\nThe reason is what lets someone later decide whether the entry "
        "still holds.")
