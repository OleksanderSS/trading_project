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

from tests.contracts._dead_config_scan import scan

#: Measured 2026-08-22, after `attention_window` was removed and the VIX period
#: was wired. Lower this when findings are fixed; never raise it.
CEILING = 21

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
