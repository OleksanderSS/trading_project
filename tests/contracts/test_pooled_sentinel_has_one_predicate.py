"""The pooled sentinel is compared in exactly one module.

`__POOLED__` is the ticker slot of a context that means "every ticker". It is
not a value in the data, so any code that compares it to a real ticker column
gets an empty result -- and an empty result is read as "no data", never as
"the filter is wrong".

That mistake has now been made three times, in three modules, by three
different routes, and each looked correct in review because each IS correct
for a real ticker:

* `walk_forward_validation` -- 159,149 rows matched none, the exception went
  to `logger.debug`, the function returned None, and the walk-forward
  stability rung was silently off for every pooled context (REGISTER #189).
* `prediction/data_preparation_service` -- the same filter returned an empty
  frame, so Stage 5 produced 0 predictions from 7 champions while the
  pipeline logged "completed successfully" (REGISTER #210, #211).
* `prediction/prediction_context_manager` -- a second copy of that same line,
  which nothing had exercised.

Three occurrences of one shape is the point at which a scanner is cheaper
than attention. The rule is not "compare carefully"; it is that only
`modeling_context` may compare, and everyone else asks it.
"""
from __future__ import annotations

import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src"

#: The one module allowed to know what the sentinel looks like.
OWNER = "pipeline/modeling_context.py"

#: A comparison against the sentinel: the constant or its literal, on either
#: side of ==, !=, .eq() or an `in` test. Docstrings and comments are stripped
#: before matching, so explaining the rule does not violate it.
COMPARISON = re.compile(
    r"(==|!=|\.eq\()\s*(POOLED_TICKER|['\"]__POOLED__['\"])"
    r"|(POOLED_TICKER|['\"]__POOLED__['\"])\s*(==|!=)"
)


def _code_only(text: str) -> str:
    """Source with comments and docstrings blanked, line count preserved."""
    without_comments = re.sub(r"#[^\n]*", "", text)
    return re.sub(
        r"(\"\"\"|''')(?:.|\n)*?\1",
        lambda m: "\n" * m.group(0).count("\n"),
        without_comments,
    )


def _offenders() -> list[str]:
    found = []
    for path in SRC.rglob("*.py"):
        rel = path.relative_to(SRC).as_posix()
        if rel == OWNER or rel.startswith("archive/"):
            continue
        code = _code_only(path.read_text(encoding="utf-8", errors="replace"))
        for number, line in enumerate(code.splitlines(), 1):
            if COMPARISON.search(line):
                found.append(f"{rel}:{number}: {line.strip()}")
    return found


def test_only_modeling_context_compares_the_pooled_sentinel():
    offenders = _offenders()
    assert not offenders, (
        "The pooled sentinel is compared outside "
        f"{OWNER}. Call `is_pooled(ticker)` or `rows_for_ticker(frame, ticker)` "
        "instead -- a literal comparison here is the shape that turned three "
        "checks off without a single error message:\n  "
        + "\n  ".join(offenders)
    )


def test_the_scanner_would_catch_the_defect_it_was_written_for():
    """A scanner that cannot fail its own case proves nothing.

    This is the line that produced REGISTER #189 and #210, verbatim.
    """
    sample = '        pooled = str(ticker).upper() == POOLED_TICKER.upper()\n'
    assert COMPARISON.search(_code_only(sample))

    second = "        ticker_df = features_df[features_df['ticker'] == ticker]\n"
    assert not COMPARISON.search(_code_only(second)), (
        "filtering by a ticker variable is legitimate; only the sentinel is not"
    )

    commented = '    # `== POOLED_TICKER` was exact and case-sensitive\n'
    assert not COMPARISON.search(_code_only(commented))
