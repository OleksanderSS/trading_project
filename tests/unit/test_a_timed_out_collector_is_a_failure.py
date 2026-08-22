"""A cancelled collector loses what it had already fetched, silently.

`asyncio.wait_for` CANCELS the coroutine it is waiting on. Collectors
accumulate rows in a local list and upsert once at the end, so one cancelled
between the fetch and the write loses everything it holds -- and the write
never runs, so there is no error from it either. A run could log "завантажено
7541 рядок" thirteen times and leave the table empty.

Per-collector timeouts removed the trigger and it has not reproduced since.
The shape is still there, so these pin the two things that keep it from being
invisible: a timeout counts as a FAILURE rather than as a quiet success, and
the message says what a cancellation costs.

The real repair is incremental persistence inside each collector. That is a
change to sixteen of them and it is not this.
"""

import inspect
import io

import pytest

from src.pipeline.stages.collection.orchestrator import CollectionStage


def _source() -> str:
    return io.open(
        "src/pipeline/stages/collection/orchestrator.py", encoding="utf-8"
    ).read()


def test_a_timeout_is_re_raised_not_swallowed():
    """Returning None would count as 'ran fine, nothing new'.

    Checked against the parsed tree, not the source text. The first version of
    this matched strings and failed on its own subject: the handler's comment
    reads "Re-raise (not return None)", so a test forbidding the text
    "return None" found it in the comment explaining why it is not there.
    """
    import ast
    import textwrap

    tree = ast.parse(textwrap.dedent(inspect.getsource(CollectionStage)))
    handlers = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ExceptHandler)
        and isinstance(node.type, ast.Name)
        and node.type.id == "TimeoutError"
    ]
    assert handlers, "no TimeoutError handler found"

    for handler in handlers:
        statements = [n for n in ast.walk(ast.Module(body=handler.body, type_ignores=[]))]
        assert any(isinstance(n, ast.Raise) for n in statements), (
            "a timeout must reach the caller as an exception"
        )
        assert not any(isinstance(n, ast.Return) for n in statements), (
            "returning from a timeout handler makes it indistinguishable from success"
        )


def test_the_timeout_message_says_the_data_is_lost():
    """"Exceeded its timeout" reads like a stall. It is a loss."""
    source = _source()
    assert "CANCELLED" in source
    assert "not yet written is lost" in source


class _Recorder:
    def __init__(self):
        self.errors: list[str] = []
        self.infos: list[str] = []

    def error(self, msg, *args, **kwargs):
        self.errors.append(msg % args if args else msg)

    def info(self, msg, *args, **kwargs):
        self.infos.append(msg % args if args else msg)

    def warning(self, msg, *args, **kwargs):
        self.infos.append(msg % args if args else msg)

    def isEnabledFor(self, _level):
        return False


class _Collector:
    def __init__(self, collector_type):
        self.collector_type = collector_type


@pytest.fixture
def stage():
    instance = CollectionStage.__new__(CollectionStage)
    instance.logger = _Recorder()
    return instance


def test_an_exception_result_is_counted_as_failed_not_silent(stage):
    """The three outcomes were once one counter, and dead sources hid in it."""
    results = [TimeoutError("took too long"), None]
    collectors = [_Collector("slow_one"), _Collector("quiet_one")]

    stage.process_and_save_results(results, collectors)

    said = " ".join(stage.logger.errors + stage.logger.infos)
    assert "slow_one" in said
    assert "TimeoutError" in said, "the type name is what makes an empty message readable"


def test_a_silent_collector_is_reported_separately_from_a_failed_one(stage):
    """Returning nothing and blowing up are different facts."""
    stage.process_and_save_results([None], [_Collector("quiet_one")])
    assert any("no new data" in line for line in stage.logger.infos)
    assert not stage.logger.errors
