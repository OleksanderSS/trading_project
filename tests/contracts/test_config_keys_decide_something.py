"""A config key that decides nothing may not be added, and the count may only fall.

A missing key fails loudly the first time something looks for it. A dead one
sits in the YAML looking like a control, gets read into an attribute, often
gets printed in a startup line, and changes nothing. It is invisible by
nature: the only way to notice is to look on purpose.

Three have cost real time here -- `daily_max_years`, `vix.params.period` and
`attention_window`. See `_dead_config_scan` for what each did.

This is a ratchet, like the silent-failure and unreachable-code contracts next
to it. 21 remain, almost all in monitoring and analytics modules that are not
on the pipeline's path. The zones that ARE on it must stay empty.
"""
from __future__ import annotations

import pytest

from tests.contracts._dead_config_scan import scan, scan_never_read

#: Measured 2026-09-08: 21 -> 16, after the three that were on the TRADING path
#: were removed -- VirtualPortfolio's stop_loss_pct and take_profit_pct (read
#: into attributes, applied nowhere, under a config comment saying "read by
#: VirtualPortfolio"), and TradingExecutionStage's exposure_monitor
#: (constructed, configured, asked nothing, labelled "✅ Integrated"). The 16
#: that remain are monitoring and analytics modules off the pipeline's path.
#: Lower this when findings are fixed; never raise it.
CEILING = 16

#: Where the batch is actually built. A setting that decides nothing here is a
#: window, a threshold or a limit that an operator believes they control and
#: does not -- which is exactly how the VIX collector came to announce 30 days
#: in its log while fetching 60.
DATA_PATH_PREFIXES = (
    "src/data/collectors/",
    "src/features/",
)


@pytest.fixture(scope="module")
def findings():
    return scan()


def test_dead_config_keys_do_not_multiply(findings):
    assert len(findings) <= CEILING, (
        f"{len(findings)} config keys decide nothing, ceiling is {CEILING}. "
        "Either read the value where it belongs, or delete the key -- do not "
        "leave a number in the config that only reaches a log line.\n"
        + "\n".join(f"  {finding}" for finding in findings)
    )


def test_nothing_on_the_data_path_is_decorative(findings):
    offenders = [
        finding for finding in findings
        if finding.path.startswith(DATA_PATH_PREFIXES)
    ]
    assert not offenders, (
        "A collector or enricher reads a setting and never uses it. This is "
        "how the VIX window came to be 60 days while the config said 30:\n"
        + "\n".join(f"  {finding}" for finding in offenders)
    )


# ---------------------------------------------------------------------------
# The other half of the family, added 2026-09-07 (#295).
#
# Everything above needs the key to be READ into an attribute first. A key that
# no Python line ever mentions is invisible to it, and there were 39 of those
# in collectors.yaml alone. Two were doing real damage:
#
#   `cache_duration_minutes`  in seventeen collector blocks, beside a
#                             `cache_ttl` in seconds that IS read. Sixteen
#                             pairs agreed. NewsAPI's did not: the readable key
#                             said refresh hourly, the live one said once a
#                             day, and once a day is what happened.
#   `intraday_max_days: 60`   the exact number the yahoo collector had already
#                             replaced with a per-interval table, because a
#                             flat 60 threw away 92% of the hourly history.
#
# Both are gone. 21 remain, and they are all in collectors.yaml, so the
# data-path rule above cannot be applied to them yet.
# ---------------------------------------------------------------------------

#: Measured 2026-09-08: 39 -> 21 -> 15 -> 9 -> ZERO, as each batch was read and
#: either wired or deleted. Every one of them was a promise to whoever edits the
#: file that something would change, and four were the same shape -- one number
#: written in two places, with the config copy inert:
#:
#:   cache_duration_minutes  in 17 blocks, beside the cache_ttl that runs.
#:                           NewsAPI's two disagreed by 24x
#:   intraday_max_days: 60   the exact figure the yahoo collector had replaced
#:                           with a per-interval table, after a flat 60 threw
#:                           away 92% of the hourly history
#:   huggingface.max_rows    10,000 declared while 999,396 rows arrived
#:   bigquery max_days /     90 and 10000, while the SQL beside them already
#:   max_rows                said INTERVAL 90 DAY and LIMIT 10000
#:
#: A ceiling of zero is strict on purpose, and there are two legitimate ways
#: past it: read the value where it belongs, or delete the key. Raising this
#: number is not one of them.
UNREAD_CEILING = 0


@pytest.fixture(scope="module")
def unread():
    return scan_never_read()


def test_config_keys_no_code_mentions_do_not_multiply(unread):
    assert len(unread) <= UNREAD_CEILING, (
        f"{len(unread)} config keys are named by no source file, ceiling is "
        f"{UNREAD_CEILING}. A setting nothing reads is a promise to whoever "
        "edits it that something will change.\n"
        + "\n".join(f"  {key}" for key in unread)
    )


def test_no_collector_declares_its_cache_window_twice(unread):
    """The seventeen-fold case, pinned so it cannot come back.

    `cache_ttl` is what BaseCollector.get_cache_ttl reads. Anything else that
    looks like a cache window beside it is decoration that contradicts it.
    """
    import yaml

    from tests.contracts._dead_config_scan import PROJECT_ROOT

    blocks = yaml.safe_load(
        (PROJECT_ROOT / "src/config/collectors.yaml").read_text(encoding="utf-8")
    )["collectors"]
    offenders = [
        name for name, block in blocks.items()
        if isinstance(block, dict) and "cache_duration_minutes" in block
    ]
    assert not offenders, (
        f"{offenders} declare a cache window in minutes beside the cache_ttl "
        "in seconds that actually runs. NewsAPI's two disagreed by a factor "
        "of 24.")


def test_the_scan_does_not_count_logging_as_use():
    """The whole point: the VIX period WAS used -- in an f-string in a log line."""
    import ast

    from tests.contracts._dead_config_scan import _inside_logging, _parents

    tree = ast.parse(
        "class C:\n"
        "    def __init__(self, configs):\n"
        "        self.period = configs.get('period', '30d')\n"
        "        self.logger.info(f'started with {self.period}')\n"
        "        self.other = configs.get('other', 1)\n"
        "        use(self.other)\n"
    )
    parents = _parents(tree)
    loads = {
        node.attr: node
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load)
        and isinstance(node.value, ast.Name) and node.value.id == "self"
        and node.attr in ("period", "other")
    }
    assert _inside_logging(loads["period"], parents), "log use counted as real use"
    assert not _inside_logging(loads["other"], parents), "real use counted as logging"
