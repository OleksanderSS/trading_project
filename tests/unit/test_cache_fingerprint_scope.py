"""The features cache invalidated itself on changes it is not protecting.

The code fingerprint gates stages 0-3 -- collection, processing, feature
engineering -- by asking whether the code that BUILT features.parquet and
targets.parquet has changed. It hashed all of src/pipeline/stages, which
also holds modeling, prediction, trading and evaluation: stages that run
after the cached artifact exists and cannot influence a single value in it.

So a Stage 4 fix moved the fingerprint. bb7faa06 touched
stages/modeling/orchestrator.py and stages/prediction/model_resolver.py,
and the next prepare run would have re-collected and re-enriched for hours
to produce a byte-identical batch.

That is not merely wasteful. An invalidation that fires when nothing it
protects has changed teaches its operator to bypass it -- and bypassing it
is exactly the failure this mechanism was built to stop, after a broken
collector cached itself in place for six days.

The direction of the rule matters. It is an EXCLUSION list, so a stage
subdirectory added later is hashed by default: over-hashing costs time,
under-hashing serves features built by code that has since been fixed.
"""
from __future__ import annotations

from pathlib import Path

from src.cli.pipeline_executor import PipelineExecutor

ROOT = Path(__file__).resolve().parents[2]


def _fingerprinted() -> set[Path]:
    return {
        path.relative_to(ROOT)
        for path in PipelineExecutor._fingerprint_files(ROOT)
    }


def _rel(*parts: str) -> Path:
    return Path("src/pipeline/stages").joinpath(*parts)


# ------------------------------------------------------- what is covered


def test_the_stages_that_build_the_artifact_are_hashed():
    covered = _fingerprinted()

    for stage in ("collection", "processing", "feature_engineering"):
        assert any(_rel(stage) in path.parents for path in covered), (
            f"{stage} builds the cached artifact and must be fingerprinted"
        )


def test_the_stage_entry_points_are_hashed():
    """stage_1..3 dispatch into those subtrees and are code in their own right."""
    covered = _fingerprinted()

    for name in ("stage_1_collection.py", "stage_2_processing.py",
                 "stage_3_feature_engineering.py", "base_stage.py"):
        assert _rel(name) in covered, name


def test_the_enrichers_and_targets_are_still_hashed():
    """The trees the earlier fixes added -- guarded against a regression in
    the exclusion logic, which walks by path prefix."""
    covered = _fingerprinted()

    for tree in ("src/features", "src/targets", "src/processing",
                 "src/data/collectors", "src/data/validation"):
        assert any(Path(tree) in path.parents for path in covered), tree


# ---------------------------------------------------- what is excluded


def test_the_later_stages_are_not_hashed():
    covered = _fingerprinted()

    for stage in PipelineExecutor._NON_CACHED_STAGE_DIRS:
        offenders = [p for p in covered if _rel(stage) in p.parents]
        assert not offenders, (
            f"{stage} runs after the cached artifact exists, so hashing it "
            f"invalidates the cache for changes that cannot affect it: "
            f"{offenders[:3]}"
        )


def test_the_two_files_that_prompted_this_are_out():
    covered = _fingerprinted()

    assert _rel("modeling", "orchestrator.py") not in covered
    assert _rel("prediction", "model_resolver.py") not in covered


def test_the_exclusion_only_applies_under_the_stages_root():
    """A directory named 'modeling' elsewhere must not be swept up by it."""
    assert not PipelineExecutor._is_non_cached_stage(
        ROOT / "src" / "features" / "modeling" / "whatever.py", ROOT
    )
    assert PipelineExecutor._is_non_cached_stage(
        ROOT / "src" / "pipeline" / "stages" / "modeling" / "whatever.py", ROOT
    )


def test_every_excluded_directory_actually_exists():
    """An exclusion naming a directory that is gone silently protects
    nothing, and reads as if it does -- the hand-maintained-list drift this
    codebase keeps producing."""
    stages_root = ROOT / "src" / "pipeline" / "stages"

    for stage in PipelineExecutor._NON_CACHED_STAGE_DIRS:
        assert (stages_root / stage).is_dir(), stage


# ------------------------------------------------------------ the effect


def test_a_stage_4_edit_does_not_move_the_fingerprint(tmp_path, monkeypatch):
    """The whole point, stated as behaviour rather than as a file list."""
    import shutil

    root = tmp_path / "repo"
    (root / "src" / "pipeline" / "stages" / "modeling").mkdir(parents=True)
    (root / "src" / "pipeline" / "stages" / "feature_engineering").mkdir(parents=True)
    (root / "src" / "pipeline" / "stages" / "modeling" / "orchestrator.py").write_text(
        "VERSION = 1", encoding="utf-8"
    )
    (root / "src" / "pipeline" / "stages" / "feature_engineering" / "e.py").write_text(
        "VERSION = 1", encoding="utf-8"
    )

    def fingerprint() -> str:
        import hashlib

        hasher = hashlib.sha256()
        for path in sorted(PipelineExecutor._fingerprint_files(root)):
            hasher.update(str(path.relative_to(root)).replace("\\", "/").encode())
            hasher.update(path.read_bytes())
        return hasher.hexdigest()

    before = fingerprint()

    (root / "src" / "pipeline" / "stages" / "modeling" / "orchestrator.py").write_text(
        "VERSION = 2", encoding="utf-8"
    )
    assert fingerprint() == before, "a Stage 4 edit still invalidates the cache"

    (root / "src" / "pipeline" / "stages" / "feature_engineering" / "e.py").write_text(
        "VERSION = 2", encoding="utf-8"
    )
    assert fingerprint() != before, "a Stage 3 edit no longer invalidates the cache"

    shutil.rmtree(root)


# --------------------------------------------------------------------------
# Helper modules the cached stages import.
#
# Only src/pipeline/stages is hashed by directory, so src/pipeline's own
# helpers were invisible -- while stages/feature_engineering/orchestrator.py
# imports is_target_like_column and targets.py imports
# is_direct_target_column from target_column_utils to decide which columns
# are features and which are targets. A change there changes what
# features.parquet and targets.parquet CONTAIN, and the fingerprint would
# not have moved.
# --------------------------------------------------------------------------


def test_the_helper_modules_are_hashed():
    covered = _fingerprinted()

    for module in PipelineExecutor._FINGERPRINT_FILES:
        assert Path(module) in covered, module


def test_the_listed_helpers_are_the_ones_the_cached_stages_import():
    """Pins the list against reality rather than against itself.

    Every src/pipeline/*.py module imported by a fingerprinted stage decides
    something about the cached artifact, so it belongs in the list. A new
    such import should fail here rather than silently go unhashed.
    """
    import re

    listed = {Path(f).name for f in PipelineExecutor._FINGERPRINT_FILES}
    imported: set[str] = set()

    for stage in ("collection", "processing", "feature_engineering"):
        stage_dir = ROOT / "src" / "pipeline" / "stages" / stage
        sources = list(stage_dir.rglob("*.py")) if stage_dir.is_dir() else []
        sources += list((ROOT / "src" / "pipeline" / "stages").glob(f"stage_*{stage}*.py"))
        for path in sources:
            text = path.read_text(encoding="utf-8", errors="replace")
            for match in re.finditer(r"from src\.pipeline\.([a-z_]+) import", text):
                imported.add(f"{match.group(1)}.py")

    missing = imported - listed
    assert not missing, (
        f"cached stages import {sorted(missing)} from src/pipeline, and those "
        f"modules are not in the code fingerprint"
    )


def test_the_helpers_do_not_drag_in_the_orchestrators():
    """src/pipeline also holds the stage 4-7 orchestrators. Naming files
    rather than the directory is what keeps them out."""
    covered = {p.as_posix() for p in _fingerprinted()}

    for excluded in ("src/pipeline/pipeline_orchestrator.py",
                     "src/pipeline/hybrid_orchestrator.py",
                     "src/pipeline/pipeline_factory.py"):
        assert excluded not in covered, excluded
