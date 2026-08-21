"""A news source that cannot be placed in time is not a news source.

`huggingface_data` held 999,396 rows of two columns -- `text` and `hash`.
Every run read all of them, spent fourteen and a half minutes filtering them
by keyword, contributed the 728,862 survivors to the news frame, and dropped
every one at deduplication for carrying no title, no timestamp and no source.
Net contribution: zero. The only trace was a warning that 762,436 news records
had been discarded, which reads like lost data rather than like a source that
was never news.

The decision is available from the schema, so it is made there.
"""

import pandas as pd
import pytest

from src.pipeline.stages.collection.orchestrator import NEWS_DATE_ALIASES


class _Manager:
    """Records which tables were actually read."""

    def __init__(self, schemas):
        self.schemas = schemas
        self.reads: list[str] = []

    def get_table_schema(self, table_name):
        return self.schemas[table_name]

    def fetch_data_from_table(self, table_name):
        self.reads.append(table_name)
        return pd.DataFrame({'text': ['x']})


class _Stage:
    """The gate under test, with the orchestrator's dependencies stubbed out."""

    from src.pipeline.stages.collection.orchestrator import (
        CollectionStage as _Real,
    )
    _news_table_can_be_dated = _Real._news_table_can_be_dated

    def __init__(self, manager, logger):
        self.db_manager = manager
        self.logger = logger


class _Logger:
    def __init__(self):
        self.warnings: list[str] = []

    def warning(self, msg, *args):
        self.warnings.append(msg % args if args else msg)


def _stage(schemas):
    manager = _Manager(schemas)
    logger = _Logger()
    return _Stage(manager, logger), manager, logger


def test_a_table_with_no_date_column_is_refused():
    stage, manager, logger = _stage({'huggingface_data': {'text': 'VARCHAR', 'hash': 'VARCHAR'}})
    assert stage._news_table_can_be_dated('huggingface_data') is False
    assert manager.reads == []                      # never read
    assert 'no publication time' in logger.warnings[0]
    assert 'huggingface_data' in logger.warnings[0]


@pytest.mark.parametrize('alias', NEWS_DATE_ALIASES)
def test_every_alias_the_renamer_accepts_passes_the_gate(alias):
    """The gate and the rename read one list, so they cannot disagree."""
    stage, _, logger = _stage({'some_news': {'title': 'VARCHAR', alias: 'VARCHAR'}})
    assert stage._news_table_can_be_dated('some_news') is True
    assert logger.warnings == []


def test_the_real_article_sources_still_pass():
    schemas = {
        'newsapi_articles': {'title': 'VARCHAR', 'publishedAt': 'VARCHAR', 'source': 'VARCHAR'},
        'google_news': {'title': 'VARCHAR', 'published_date': 'VARCHAR', 'source': 'VARCHAR'},
        'rss_news': {'title': 'VARCHAR', 'published_date': 'VARCHAR', 'source': 'VARCHAR'},
    }
    stage, _, _ = _stage(schemas)
    for table in schemas:
        assert stage._news_table_can_be_dated(table) is True, table


def test_sec_filings_are_refused_over_one_letter_of_case():
    """Pins the open defect rather than pretending it is fixed.

    `sec_filings` carries `filingDate`; the alias list carries `filing_date`.
    24,365 dated, ticker-tagged filings are discarded over the capital D, and
    have been counted into a lump warning about 762,436 "lost news records"
    that hid which source they came from.

    They are not admitted yet on purpose: the fields available -- `form`,
    `primaryDocDescription` -- are codes like "10-Q", not prose, and feeding
    them to the sentiment model would manufacture a reading rather than
    recover one. Fixing the mapping means routing filings as events, which is
    a decision, not a rename. See #66 in docs/REGISTER.md.
    """
    stage, _, logger = _stage({
        'sec_filings': {
            'accessionNumber': 'VARCHAR', 'filingDate': 'VARCHAR',
            'reportDate': 'VARCHAR', 'form': 'VARCHAR', 'ticker': 'VARCHAR',
        },
    })
    assert stage._news_table_can_be_dated('sec_filings') is False
    assert 'filingDate' in logger.warnings[0]


def test_report_date_would_be_a_look_ahead_and_is_not_an_alias():
    """`reportDate` is the period covered; `filingDate` is when it became public."""
    assert 'reportDate' not in NEWS_DATE_ALIASES


def test_an_unreadable_schema_fails_open():
    """Never lose a source to a failure to inspect it."""

    class _Broken(_Manager):
        def get_table_schema(self, table_name):
            raise RuntimeError('table locked')

    stage = _Stage(_Broken({}), _Logger())
    assert stage._news_table_can_be_dated('anything') is True
    assert 'admitting it' in stage.logger.warnings[0]


def test_sec_filings_are_no_longer_classified_as_news():
    """They are events. Filed as news they were dropped whole, every run."""
    from src.pipeline.stages.collection.orchestrator import classify_source_table

    assert classify_source_table('sec_filings', {'type': 'sec_filings'}) == 'corporate_filings'
    assert classify_source_table('google_news', {'type': 'google_news'}) == 'news'
    assert classify_source_table('newsapi_articles', {'type': 'newsapi'}) == 'news'


def test_the_filings_enricher_is_registered_and_loadable():
    """A source with no enricher wired to it is a source that does nothing."""
    import importlib
    import io as _io

    import yaml

    cfg = yaml.safe_load(_io.open('src/config/enrichment.yaml', encoding='utf-8'))

    def _find(node, key):
        if isinstance(node, dict):
            if key in node:
                return node[key]
            for value in node.values():
                found = _find(value, key)
                if found is not None:
                    return found
        return None

    entry = _find(cfg, 'corporate_filings')
    assert entry is not None, 'corporate_filings not registered in enrichment.yaml'
    module = importlib.import_module(entry['module'])
    enricher = getattr(module, entry['class'])(entry.get('params', {}))
    assert enricher.name == 'corporate_filings'
