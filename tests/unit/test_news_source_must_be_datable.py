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


# ---------------------------------------------------------------------------
# One list, five copies. Added 2026-09-07 (#294).
#
# "Which column means when this happened" was answered by five separate lists
# and no two agreed. The one that decides ADMISSION (NEWS_DATE_ALIASES) was
# fine. The one that decides the UTC CONVERSION was not: it knew neither
# `publishedAt` nor `published_date`, which are the only time columns the three
# real news tables have, so `_normalize_data` converted nothing for any of
# them and their dates stayed strings. It also preferred `created_at` -- when
# the row was written -- over `published_at`.
# ---------------------------------------------------------------------------

from src.features.utils.datetime_utils import (  # noqa: E402
    TIME_COLUMNS, first_time_column,
)

#: The columns each real news table actually has, read off the database on
#: 2026-09-07. Hardcoded on purpose: the test must fail if the pipeline stops
#: understanding these names, not quietly follow the schema wherever it goes.
REAL_NEWS_SCHEMAS = {
    'newsapi_articles': ['source', 'author', 'title', 'description', 'url',
                         'urlToImage', 'publishedAt', 'content',
                         'search_term', 'hash'],
    'google_news': ['title', 'link', 'published_date', 'source', 'content',
                    'hash'],
    'rss_news': ['title', 'link', 'published_date', 'source', 'content',
                 'hash'],
}


@pytest.mark.parametrize('table,columns', sorted(REAL_NEWS_SCHEMAS.items()))
def test_the_utc_conversion_finds_a_date_on_every_real_news_table(table, columns):
    """`_normalize_data` skipped the conversion entirely on all three."""
    found = first_time_column(columns)
    assert found is not None, (
        f"{table} carries {columns} and no time column was recognised, so "
        "_normalize_data converts nothing and the dates stay strings.")
    assert found in ('publishedAt', 'published_date'), found


def test_a_publication_time_beats_the_row_write_time():
    """`created_at` came first in the old list. That is the shape of #286."""
    assert first_time_column(
        ['created_at', 'published_at', 'title']) == 'published_at'
    assert first_time_column(
        ['created_at', 'publishedAt']) == 'publishedAt'


def test_we_never_treat_our_own_fetch_time_as_the_event_time():
    """`collected_at` is when WE fetched the row. It must never be in here."""
    assert 'collected_at' not in TIME_COLUMNS


def test_a_precise_timestamp_is_preferred_over_a_bare_date():
    """A date has no time of day and can place an event on the wrong bar."""
    assert first_time_column(['date', 'timestamp']) == 'timestamp'


def test_the_admission_list_stays_a_subset_of_the_shared_one():
    """NEWS_DATE_ALIASES is a strict subset by design, not a rival copy.

    It is stricter because admission demands a PUBLICATION time. If a name is
    added to one and not the other, a source can be admitted and then never
    normalised -- which is the defect this whole block exists for.
    """
    # `filing_date` is the SEC family's publication time and belongs only to
    # the admission list; everything else must be a name the shared list knows.
    stray = set(NEWS_DATE_ALIASES) - set(TIME_COLUMNS) - {'filing_date'}
    assert not stray, (
        f"{sorted(stray)} admit a news table but are unknown to "
        "datetime_utils.TIME_COLUMNS, so those rows would be admitted and "
        "then left unconverted.")


def test_the_enricher_and_the_converter_read_the_same_list():
    """Four modules declared this list. Three of them now import one."""
    from src.features.enrichers import keyword_entity_enricher

    assert keyword_entity_enricher.TIME_COLUMNS is TIME_COLUMNS, (
        "the keyword enricher has its own copy again. It was the only list "
        "that knew `publishedAt`, and being alone in knowing it is how the "
        "conversion came to be skipped everywhere else.")


def test_huggingface_is_still_refused_after_all_of_this():
    """The widening must not readmit the wikitext dump."""
    assert first_time_column(['text', 'hash']) is None
    stage, _, _ = _stage({'huggingface_data': {'text': 'VARCHAR',
                                               'hash': 'VARCHAR'}})
    assert stage._news_table_can_be_dated('huggingface_data') is False
