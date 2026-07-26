"""Regression test for CROSS_DOMAIN_PROPAGATION's target_domains invariant.

CROSS_DOMAIN_PROPAGATION (dean_os/analyst_core/cross_domain_signal_bus.py)
is a hand-maintained dict whose target_domains lists must match real
domain_ids from dean_os.domain_profiles.list_domain_ids() -- there was no
test locking this in, and it had already drifted once (used "industrial",
"consumer", "financials", none of which are real domain_ids) before being
fixed this session. from_signal_bus() (artifact_evidence_loader.py)
silently drops any signal whose domain doesn't match a real one, so a typo
here is a silent no-op, not a crash -- exactly the kind of drift that goes
unnoticed without a test.
"""
from __future__ import annotations

from dean_os.analyst_core.cross_domain_signal_bus import CROSS_DOMAIN_PROPAGATION
from dean_os.domain_profiles import list_domain_ids


def test_all_target_domains_are_real_registered_domain_ids():
    real_domain_ids = set(list_domain_ids())
    for event_class, rule in CROSS_DOMAIN_PROPAGATION.items():
        bogus = set(rule["target_domains"]) - real_domain_ids
        assert not bogus, (
            f"CROSS_DOMAIN_PROPAGATION[{event_class!r}].target_domains "
            f"references non-existent domain_id(s) {bogus} -- must be one "
            f"of {sorted(real_domain_ids)}"
        )
