"""Path containment is what stands between a config value and the filesystem.

validate_safe_path guards every FileManager read and write, and the .env
search. Three things were wrong with it:

1. The symlink check ran on the RESOLVED path:

       target_path = Path(path).resolve()
       if not allow_symlinks and target_path.is_symlink():

   resolve() follows symlinks, so it asked whether the already-resolved path
   is a link, which by construction it is not. The flag never fired, and no
   caller passes it.

   Containment itself was never at risk: resolve() turns a link pointing
   outside the project into the outside path, and relative_to then rejects
   it. The defect is a promise not kept, not an escape route -- and the tests
   below assert both halves of that.

2. A relative path was resolved against the process working directory rather
   than base_dir, so the same argument meant different files depending on
   where the program was started.

3. OSError escaped: on Windows resolve() raises it for names the filesystem
   rejects, instead of the documented PathValidationError.

Merged with the original unittest suite that lived in this file, whose
test_symlink_denied called self.skipTest() unconditionally -- the one test
that would have caught the inert flag never ran at all.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from src.core.security.path_validator import PathValidationError, validate_safe_path


@pytest.fixture()
def base(tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "file.txt").write_text("inside", encoding="utf-8")
    return tmp_path


def test_a_path_inside_the_base_is_accepted(base):
    result = validate_safe_path(base / "data" / "file.txt", base_dir=base)
    assert result == (base / "data" / "file.txt").resolve()


def test_a_path_outside_the_base_is_rejected(base, tmp_path_factory):
    outside = tmp_path_factory.mktemp("elsewhere") / "secret.txt"
    outside.write_text("out", encoding="utf-8")

    with pytest.raises(PathValidationError, match="outside authorized"):
        validate_safe_path(outside, base_dir=base)


@pytest.mark.parametrize("traversal", [
    "../escape.txt",
    "data/../../escape.txt",
    "data/./../../escape.txt",
])
def test_traversal_sequences_are_rejected(base, traversal):
    with pytest.raises(PathValidationError):
        validate_safe_path(traversal, base_dir=base)


def test_a_relative_path_is_resolved_against_the_base_not_the_cwd(base, monkeypatch):
    """The same argument used to mean different files depending on where the
    process was started."""
    elsewhere = base.parent / "cwd_somewhere"
    elsewhere.mkdir(exist_ok=True)
    monkeypatch.chdir(elsewhere)

    assert validate_safe_path("data/file.txt", base_dir=base) == (
        base / "data" / "file.txt"
    ).resolve()


def test_a_malformed_name_raises_the_documented_error(base):
    with pytest.raises(PathValidationError):
        validate_safe_path("data/a\x00b.txt", base_dir=base)


def test_the_base_directory_itself_is_valid(base):
    assert validate_safe_path(base, base_dir=base) == base.resolve()


def _make_symlink(link: Path, target: Path) -> bool:
    try:
        os.symlink(target, link)
        return True
    except (OSError, NotImplementedError):
        return False


def test_a_symlink_pointing_outside_is_rejected(base, tmp_path_factory):
    """This is the containment guarantee, and it held even while the
    allow_symlinks flag was inert."""
    outside = tmp_path_factory.mktemp("elsewhere") / "secret.txt"
    outside.write_text("out", encoding="utf-8")
    link = base / "data" / "sneaky.txt"

    if not _make_symlink(link, outside):
        pytest.skip("creating symlinks requires privileges not held here")

    with pytest.raises(PathValidationError):
        validate_safe_path(link, base_dir=base)


def test_a_symlink_staying_inside_is_rejected_unless_allowed(base):
    """What the inert flag failed to do: resolve() keeps such a link inside
    the base, so the containment check passes it."""
    link = base / "data" / "alias.txt"

    if not _make_symlink(link, base / "data" / "file.txt"):
        pytest.skip("creating symlinks requires privileges not held here")

    with pytest.raises(PathValidationError, match="Symlinks are not allowed"):
        validate_safe_path(link, base_dir=base)

    assert validate_safe_path(link, base_dir=base, allow_symlinks=True)


def test_the_symlink_check_looks_at_the_unresolved_path():
    """Deterministic proof that does not need symlink privileges: the guard
    is handed the path as given, so a link ANYWHERE along it is seen. Run
    against the resolved path -- as the old code did -- this could never
    report True."""
    import inspect

    from src.core.security import path_validator

    source = inspect.getsource(path_validator.validate_safe_path)
    code = "\n".join(
        line for line in source.splitlines() if not line.strip().startswith("#")
    )

    assert "_reject_symlinks_within(base_path, given)" in code, (
        "the symlink guard must receive the unresolved path"
    )
    assert "target_path.is_symlink()" not in code, (
        "checking is_symlink() on a resolved path can never be True"
    )


def test_a_sibling_directory_sharing_the_base_name_is_rejected(base):
    """Kept from the original unittest suite: an earlier version used
    str(target).startswith(str(base)), which treats '.../data_secret/x' as
    contained in '.../data'. relative_to() is the correct check."""
    sibling = Path(f"{base}_secret")
    sibling.mkdir(exist_ok=True)
    leaked = sibling / "leaked.txt"
    leaked.touch()
    try:
        with pytest.raises(PathValidationError):
            validate_safe_path(leaked, base_dir=base)
    finally:
        leaked.unlink()
        sibling.rmdir()
