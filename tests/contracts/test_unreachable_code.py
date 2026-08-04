"""No statement in src/ may sit where control flow can never reach it.

This ratchet exists because of a regression I introduced. ModelingStage.__init__
ended with three lines:

    self.models_dir = self.config_manager.get_models_path()
    self.diary_path = Path(...)
    self._init_infrastructure()

Commit a4ca9176 inserted a _resolve_test_size method above them. The three
lines ended up after that method's `return`, inside its except block, and
became unreachable. Import worked. Every unit test passed. The class simply
never gained two attributes.

The cost only appeared in a full run, on 2026-08-04: all 22 tickers raised
`'ModelingStage' object has no attribute 'diary_path'` at the moment the
champion was about to be written, the per-ticker handler caught it, and
Stage 4 finished having logged ZERO champions across 44 training runs while
the pipeline reported success.

Python does not warn about this shape, and neither pyflakes nor py_compile
treats it as an error -- it is syntactically valid. Only an AST walk sees it.
"""
from __future__ import annotations

import textwrap

import pytest

from tests.contracts._unreachable_scan import scan

#: Zero, and it should stay zero. Unlike the silent-failure and formula
#: ratchets, there is no defensible reason for a statement that cannot run.
CEILING = 0


def test_no_unreachable_statements_in_src():
    found = scan()

    assert len(found) <= CEILING, (
        f"unreachable statement(s) rose from {CEILING} to {len(found)}:\n"
        + "\n".join(f"    {finding}" for finding in found)
        + "\n\nCode after a return/raise, or after a try whose body and every "
        "handler exit, never executes. If it was meant to run, it is in the "
        "wrong block."
    )


def _write(tmp_path, name, body):
    root = tmp_path / "src"
    root.mkdir(exist_ok=True)
    (root / name).write_text(textwrap.dedent(body), encoding="utf-8")
    return root


def test_it_catches_the_exact_shape_that_caused_the_regression(tmp_path):
    """A try/except where BOTH branches return. The first version of this
    scanner missed precisely this -- it only looked for a bare `return`, and
    a Try node is not a Return, so it reported the sample clean."""
    root = _write(tmp_path, "regression.py", """
        class Stage:
            def _resolve(self):
                try:
                    return 1
                except Exception:
                    return 2
                self.diary_path = "never assigned"
    """)

    found = scan(root)

    assert len(found) == 1
    assert "diary_path" in found[0].text


def test_it_catches_a_plain_statement_after_return(tmp_path):
    root = _write(tmp_path, "plain.py", """
        def f():
            return 1
            print("dead")
    """)

    assert len(scan(root)) == 1


def test_a_try_whose_handler_falls_through_is_not_flagged(tmp_path):
    """`except: pass` lets control continue, so what follows is reachable."""
    root = _write(tmp_path, "clean.py", """
        def f(x):
            try:
                return 1
            except Exception:
                pass
            return 2
    """)

    assert scan(root) == []


def test_an_if_without_an_else_is_not_flagged(tmp_path):
    """A missing else always falls through."""
    root = _write(tmp_path, "branch.py", """
        def f(x):
            if x:
                return 1
            return 2
    """)

    assert scan(root) == []


def test_an_if_whose_branches_both_exit_is_flagged(tmp_path):
    root = _write(tmp_path, "both.py", """
        def f(x):
            if x:
                return 1
            else:
                return 2
            print("dead")
    """)

    assert len(scan(root)) == 1


def test_a_placeholder_after_return_is_not_flagged(tmp_path):
    """`...` or a string is a stylistic marker, not a lost statement."""
    root = _write(tmp_path, "stub.py", '''
        def f():
            return 1
            ...
    ''')

    assert scan(root) == []


def test_the_modeling_stage_gains_the_attributes_it_lost():
    """The regression itself, checked on the real class rather than by shape.

    _init_infrastructure creates the models directory and seeds the diary
    CSV; neither happened while these lines were stranded.
    """
    import inspect

    from src.pipeline.stages.modeling.orchestrator import ModelingStage

    init_source = inspect.getsource(ModelingStage.__init__)

    assert "self.diary_path" in init_source
    assert "self.models_dir" in init_source
    assert "self._init_infrastructure()" in init_source

    resolve_source = inspect.getsource(ModelingStage._resolve_test_size)
    assert "self.diary_path" not in resolve_source, (
        "the stranded lines are back inside _resolve_test_size"
    )
