"""The .env parser decides whether every API key in the project exists.

The project ships its own parser rather than depending on python-dotenv. It
handled the simple `KEY=value` case and quietly mangled several ordinary ones.
None of them are present in the current .env (checked: 7 plain lines, no
export prefixes, no inline comments, no quotes) -- these are the shapes that
appear the moment someone edits the file, which is exactly what adding
Telegram credentials will involve.

  export KEY=value    stored under the name "export KEY", so the collector
                      looking for KEY saw nothing: set, and still missing
  KEY=value # note    the comment became part of the secret, which surfaces
                      later as an unexplained 401 from the provider
  KEY="value"         .strip('"') removed every quote character, not one
                      matched pair
  =value              os.environ[''] rather than a skipped line

Also: mask_secret revealed the first and last four characters of anything
longer than 8, so a 9-character secret was logged 8 characters exposed.
"""
from __future__ import annotations

import logging

import pytest

from src.core.security.secure_secrets_manager import (
    SecretsManager,
    load_dotenv,
    mask_secret,
)


@pytest.fixture()
def env_file(tmp_path, monkeypatch):
    """Write a .env inside the project root -- the loader's path guard
    (validate_safe_path) refuses anything outside it, and that guard works."""
    from src.utils.path_safety import get_path_safety

    root = get_path_safety().get_project_root()
    target = root / "tests" / "_tmp_env_fixture"
    target.mkdir(parents=True, exist_ok=True)
    path = target / ".env.fixture"

    def write(content: str) -> str:
        path.write_text(content, encoding="utf-8")
        return str(path.relative_to(root))

    yield write

    path.unlink(missing_ok=True)
    target.rmdir()


def _load(write, content, monkeypatch):
    for key in ("FIXTURE_KEY", "OTHER_KEY"):
        monkeypatch.delenv(key, raising=False)
    relative = write(content)
    return load_dotenv(relative)


def test_a_plain_assignment_loads(env_file, monkeypatch):
    import os
    _load(env_file, "FIXTURE_KEY=abc123\n", monkeypatch)
    assert os.environ["FIXTURE_KEY"] == "abc123"


def test_an_export_prefix_is_stripped(env_file, monkeypatch):
    """Otherwise the variable lands under the name "export FIXTURE_KEY"."""
    import os
    _load(env_file, "export FIXTURE_KEY=abc123\n", monkeypatch)

    assert os.environ.get("FIXTURE_KEY") == "abc123"
    assert "export FIXTURE_KEY" not in os.environ


def test_an_inline_comment_is_not_part_of_the_secret(env_file, monkeypatch):
    import os
    _load(env_file, "FIXTURE_KEY=abc123 # telegram bot\n", monkeypatch)
    assert os.environ["FIXTURE_KEY"] == "abc123"


def test_a_hash_inside_the_value_survives(env_file, monkeypatch):
    """Only ' #' starts a comment; '#' can legitimately appear in a token."""
    import os
    _load(env_file, "FIXTURE_KEY=abc#123\n", monkeypatch)
    assert os.environ["FIXTURE_KEY"] == "abc#123"


@pytest.mark.parametrize("line,expected", [
    ('FIXTURE_KEY="abc123"', "abc123"),
    ("FIXTURE_KEY='abc123'", "abc123"),
    ('FIXTURE_KEY=abc"123', 'abc"123'),
    ('FIXTURE_KEY="abc # not a comment"', "abc # not a comment"),
])
def test_one_matched_pair_of_quotes_is_removed(env_file, monkeypatch, line, expected):
    import os
    _load(env_file, line + "\n", monkeypatch)
    assert os.environ["FIXTURE_KEY"] == expected


def test_a_value_containing_equals_is_kept_whole(env_file, monkeypatch):
    import os
    _load(env_file, "FIXTURE_KEY=a=b=c\n", monkeypatch)
    assert os.environ["FIXTURE_KEY"] == "a=b=c"


def test_a_nameless_line_is_reported_and_skipped(env_file, monkeypatch, caplog):
    with caplog.at_level(logging.WARNING):
        loaded = _load(env_file, "=orphan\nFIXTURE_KEY=fine\n", monkeypatch)

    assert loaded == ["FIXTURE_KEY"]
    assert any("no variable name" in r.getMessage() for r in caplog.records)


def test_comments_and_blank_lines_are_skipped(env_file, monkeypatch):
    loaded = _load(env_file, "# a comment\n\nFIXTURE_KEY=v\n\n", monkeypatch)
    assert loaded == ["FIXTURE_KEY"]


@pytest.mark.parametrize("secret", ["", None])
def test_masking_an_absent_secret(secret):
    assert mask_secret(secret) == "None"


@pytest.mark.parametrize("length", [1, 8, 9, 12, 15])
def test_short_secrets_are_masked_completely(length):
    """first4...last4 on a 9-character secret exposes 8 of its 9 characters."""
    assert mask_secret("x" * length) == "****"


def test_long_secrets_show_only_their_edges():
    secret = "abcd" + "y" * 24 + "wxyz"
    masked = mask_secret(secret)

    assert masked == "abcd...wxyz"
    assert secret not in masked
    assert len(masked) < len(secret)


def test_the_masking_threshold_is_not_lowered_silently():
    assert SecretsManager._MASK_MIN_LENGTH >= 16
