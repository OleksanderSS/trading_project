"""The unit suite may not get worse, and a new failure fails the build.

WHY, MEASURED 2026-09-06 (REGISTER #283). `tests/unit` held 27 failures
against 3,337 passes while CI reported green: the blocking step covers
`tests/contracts` only, and the wide run sits behind `|| true`. What was going
red to nobody was not incidental --

    test_collector_identity_keys   sec_fundamentals keys on MEASUREMENTS, so
                                   "the row changes, the hash changes, dedup
                                   stores a second copy". That is #142 exactly,
                                   the defect that duplicated FRED 4.99x and
                                   added 85,478 rows a day, sitting in a second
                                   collector and announced for weeks.
    test_unbounded_join_ratchet    a RATCHET, breached: 24 sites against a
                                   ledger of 22.
    test_macro_point_in_time       the FRED hash does not change when the
                                   vintage does.

WHY A LEDGER OF IDS AND NOT A COUNT. A count lets one failure be fixed while
another appears and reports no change. The set cannot: any id that is not in
the ledger fails the build, whatever the total.

WHY THIS AND NOT "MAKE IT BLOCKING NOW". Blocking on zero turns CI red today
and keeps it red until 23 unrelated repairs land, which is how a gate gets
switched off (the `|| true` in this very workflow, #181). A ledger keeps CI
GREEN today and still fails the build on the twenty-fourth failure -- which is
the property that was actually missing.

The ledger may only shrink. Fixing a test and leaving it in the ledger is
reported as a gain to lock in, exactly like
`tests/unit/test_unbounded_join_ratchet.py` does for its own sites.

    python scripts/maintenance/unit_failure_ratchet.py
    python scripts/maintenance/unit_failure_ratchet.py --baseline   # rewrite
"""
from __future__ import annotations

import argparse
import io
import re
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LEDGER = PROJECT_ROOT / "tests" / "known_unit_failures.txt"
SUITE = "tests/unit"

FAILED = re.compile(r"^FAILED\s+(\S+)")


def _read_ledger() -> set[str]:
    if not LEDGER.exists():
        return set()
    return {
        line.strip()
        for line in io.open(LEDGER, encoding="utf-8").read().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def _run() -> set[str]:
    done = subprocess.run(
        [sys.executable, "-m", "pytest", SUITE, "-q", "--tb=no", "-rf"],
        cwd=str(PROJECT_ROOT), capture_output=True, text=True,
        encoding="utf-8", errors="replace",
    )
    output = (done.stdout or "") + (done.stderr or "")
    failures = {match.group(1) for line in output.splitlines()
                if (match := FAILED.match(line.strip()))}
    if "passed" not in output and "failed" not in output:
        raise SystemExit(
            "pytest produced no summary at all -- the suite did not run, and "
            "an empty failure set would read as a clean one:\n" + output[-2000:]
        )
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", action="store_true",
                        help="rewrite the ledger from the current run")
    args = parser.parse_args()

    failures = _run()
    known = _read_ledger()

    if args.baseline:
        header = [
            "# Known failures in tests/unit, one id per line (REGISTER #283).",
            "#",
            "# This file is a CEILING, not a to-do list: any failure NOT listed",
            "# here fails the build. It may only shrink. Rewriting it with",
            "# --baseline to make a new failure go away is the one use that",
            "# defeats its purpose -- fix the test or the code instead.",
            "#",
            f"# Recorded from a full run of {SUITE}.",
            "",
        ]
        LEDGER.write_text("\n".join(header + sorted(failures)) + "\n",
                          encoding="utf-8")
        print(f"ledger rewritten with {len(failures)} known failures")
        return 0

    new = sorted(failures - known)
    fixed = sorted(known - failures)

    print(f"{len(failures)} failing, {len(known)} in the ledger")
    if fixed:
        print(f"\n{len(fixed)} now PASS and are still in the ledger -- remove "
              f"them to lock the gain in:")
        for name in fixed:
            print(f"    {name}")

    if new:
        print(f"\n{len(new)} NEW failure(s), not in the ledger:")
        for name in new:
            print(f"    {name}")
        print("\nThe unit suite got worse. Fix it, or -- if the test itself is "
              "wrong --\nfix the test; do not rebaseline, which is how the "
              "previous 23 became invisible.")
        return 1

    print("\nNo new failures.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
