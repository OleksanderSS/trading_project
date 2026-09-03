"""No builder may grant a capability, or call itself ready, on an empty world.

Three separate defects of this shape were found on 2026-08-13: a frozen collector
snapshot reporting `ok`, a staged workbench review reporting `review_ready` after
reading zero blocks, and PipelineBridge handing the agents `verdict="clear"` for a
Stage 7 result containing nothing. Each was found by accident, one at a time.

An earlier attempt to find the rest by calling every builder and inspecting its
status did not work: the builders' default paths point at real artifacts on disk,
so that probe measured the current state of the repository rather than the
empty-input property. The fix is hermeticity -- these tests run each builder with
the working directory pointed at an empty temporary tree, so every relative
default path resolves to nothing and the builder genuinely sees no input.

Discovery is dynamic so a newly added builder is covered without editing this
file.
"""
from __future__ import annotations

import importlib
import inspect
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Builders that walk thousands of artifacts or shell out; they are covered by
# their own tests and would dominate the runtime of this one.
SKIP = {
    "dean_os.pipeline_control.pipeline_control_evidence_inventory",
    "dean_os.pipeline_control.pipeline_control_metric_artifact_materializer",
    "dean_os.review_only_automation_run",
    "dean_os.current_architecture_map",
}

# Words that claim readiness or health. A status containing one of these, with no
# qualifier saying what is absent, is a positive answer.
POSITIVE = ("ready", "clear", "passed", "verified", "complete", "healthy", "success", "authorized")
QUALIFIER = (
    "blocked", "missing", "insufficient", "unavailable", "not_", "no_", "fail", "caution",
    "waiting", "skipped", "empty", "absent", "unverifiable", "needs_", "unknown", "error",
    "partial", "pending", "stale", "warning", "gap", "unreadable", "quarantine", "limitation",
)


def _claims_readiness(status: str) -> bool:
    text = status.lower()
    if any(word in text for word in QUALIFIER):
        return False
    return any(word in text for word in POSITIVE)


def _discover() -> list[str]:
    """Module names only, found by globbing.

    Importing every dean_os module at collection time broke pytest's output
    capture and aborted the session before a single test ran, so discovery stays
    import-free and each test imports the one module it needs.
    """
    found = []
    for path in sorted(PROJECT_ROOT.glob("dean_os/**/*.py")):
        if {"__pycache__", "draft", "archive_v1"} & set(path.parts) or path.stem.startswith("__"):
            continue
        if path.stat().st_size < 400:
            continue
        dotted = ".".join(path.relative_to(PROJECT_ROOT).with_suffix("").parts)
        if dotted not in SKIP:
            found.append(dotted)
    return found


def _builders_in(dotted: str):
    """Classes in this module whose build() and __init__ both need no arguments."""
    try:
        module = importlib.import_module(dotted)
    except Exception:
        return []
    out = []
    for name, obj in vars(module).items():
        if not (inspect.isclass(obj) and getattr(obj, "__module__", "") == dotted):
            continue
        build = getattr(obj, "build", None)
        if build is None or not callable(build):
            continue
        try:
            if _has_required(inspect.signature(build)) or _has_required(inspect.signature(obj.__init__)):
                continue
        except (TypeError, ValueError):
            continue
        out.append((name, obj))
    return out


def _has_required(sig) -> bool:
    return any(
        p.default is p.empty and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
        for p in list(sig.parameters.values())[1:]
    )


MODULES = _discover()

# Builders that answer positively to an empty world and have not yet been judged
# individually. Several are probably legitimate -- a "policy packet" declaring what
# its analyst type is permitted to do, or a fixture validator whose whole subject is
# synthetic fixtures, is not claiming to have read evidence. Deciding that requires
# reading each one properly, and a rushed verdict here is worse than an honest
# "unreviewed".
#
# These are strict xfails: fixing one makes its test pass, which fails the run and
# forces its removal from this list. Closing a gap is a deletion.
UNREVIEWED = {
    "dean_os.analyst_core.domain_analyst_profile_policy_packet":
        "grants can_create_analyst_* -- likely a static policy declaration, not an evidence-backed grant",
    "dean_os.analyst_core.domain_analyst_regime_scenario_packet":
        "unreviewed",
    "dean_os.analyst_core.domain_analyst_template_decision_packet":
        "unreviewed",
    "dean_os.analyst_core.domain_analyst_vertical_slice_run":
        "unreviewed",
    "dean_os.pipeline_control.pipeline_control_metric_fixture_validation":
        "reports synthetic_fixture_control_flow_passed -- its subject really is fixtures",
    "dean_os.stress.test_phase8":
        "module is named test_* inside dean_os and is collected as a test module in its own right",
}


def _run_in_empty_world(cls, tmp_path: Path):
    init_sig = inspect.signature(cls.__init__)
    instance = cls(output_dir=str(tmp_path / "out")) if "output_dir" in init_sig.parameters else cls()
    build_sig = inspect.signature(cls.build)
    kwargs = {"save": False} if "save" in build_sig.parameters else {}
    return instance.build(**kwargs)


def _statuses(payload: dict) -> list[tuple[str, str]]:
    out = []
    summary = payload.get("summary")
    if isinstance(summary, dict):
        for key, value in summary.items():
            if isinstance(value, str) and ("status" in key or key == "verdict" or key.endswith("_verdict")):
                out.append((key, value))
    for key in ("status", "verdict"):
        value = payload.get(key)
        if isinstance(value, str):
            out.append((key, value))
    return out


def _capabilities(payload: dict) -> dict[str, bool]:
    summary = payload.get("summary")
    if not isinstance(summary, dict):
        return {}
    return {
        key: value
        for key, value in summary.items()
        if isinstance(value, bool) and (key.startswith("can_") or key.endswith("_authorized"))
    }


def test_modules_were_discovered():
    """Guards the whole file: a discovery bug would make every test below vacuous."""
    assert len(MODULES) >= 100


def _short_id(dotted: str) -> str:
    """Windows caps path length, and tmp_path is named after the test id.

    Full dotted module names pushed the per-test temp directory past the limit and
    every affected test errored in fixture setup rather than running.
    """
    return dotted.rsplit(".", 1)[-1][:40]


@pytest.mark.parametrize(
    "dotted",
    [
        pytest.param(
            name,
            id=_short_id(name),
            marks=pytest.mark.xfail(strict=True, reason=UNREVIEWED[name]),
        )
        if name in UNREVIEWED
        else pytest.param(name, id=_short_id(name))
        for name in MODULES
    ],
)
def test_builders_refuse_an_empty_world(dotted, tmp_path, monkeypatch):
    builders = _builders_in(dotted)
    if not builders:
        pytest.skip("no zero-argument builder in this module")

    empty = tmp_path / "empty_world"
    empty.mkdir()
    monkeypatch.chdir(empty)

    granted_by: dict[str, list[str]] = {}
    claimed_by: dict[str, list[tuple[str, str]]] = {}
    exercised = 0

    for name, cls in builders:
        try:
            payload = _run_in_empty_world(cls, tmp_path)
        except Exception:
            # Refusing loudly is an acceptable answer to an empty world.
            continue
        if not isinstance(payload, dict):
            continue
        exercised += 1
        granted = sorted(key for key, value in _capabilities(payload).items() if value is True)
        if granted:
            granted_by[name] = granted
        claimed = [(key, value) for key, value in _statuses(payload) if _claims_readiness(value)]
        if claimed:
            claimed_by[name] = claimed

    if not exercised:
        pytest.skip("every builder in this module refused the empty world outright")

    assert granted_by == {}, (
        f"{dotted}: {granted_by} granted with no input available. "
        "A capability must rest on evidence that was actually read."
    )
    assert claimed_by == {}, (
        f"{dotted}: {claimed_by} reported with no input available. "
        "Absence of evidence must not read as evidence of readiness."
    )
