"""The insider screener must be asked about OUR companies.

The configured URL carries `s=` empty -- no symbol filter -- with
`cnt=100&page=1`, so every run fetched the hundred most recent filings across
the WHOLE MARKET. The `tickers` argument was accepted by the caller and never
used.

That is a coverage ceiling rather than an error, and it is why the feature
looked broken twice: of 1,395 accumulated filings only 11 concern our 22
companies, so `insider_net_value_30d` reaches 5% of bars. No downstream fix
could have helped -- the rows were never collected.
"""

from urllib.parse import parse_qs, urlsplit

import pytest

from src.data.collectors.insider_collector import InsiderCollector

TEMPLATE = (
    'http://openinsider.com/screener?s=&o=&pl=&ph=&ll=&lh=&fd=0&fdr=&td=0&tdr='
    '&daysago=&xp=1&xs=1&sortcol=0&cnt=100&page=1'
)

build = InsiderCollector._urls_for_tickers


def _symbol(url):
    return parse_qs(urlsplit(url).query, keep_blank_values=True)['s'][0]


def test_one_query_per_ticker():
    urls = build([TEMPLATE], ['AAPL', 'MSFT', 'XOM'])
    assert len(urls) == 3
    assert sorted(_symbol(u) for u in urls) == ['AAPL', 'MSFT', 'XOM']


def test_the_symbol_actually_reaches_the_query():
    """The defect exactly: `s` stayed empty and the screener returned the
    market."""
    for url in build([TEMPLATE], ['AAPL']):
        assert _symbol(url) == 'AAPL'
        assert 's=&' not in url


def test_tickers_are_normalised_and_deduplicated():
    urls = build([TEMPLATE], [' aapl ', 'AAPL', 'aapl', 'msft'])
    assert sorted(_symbol(u) for u in urls) == ['AAPL', 'MSFT']


def test_the_rest_of_the_query_is_left_alone():
    """The window and the row count stay a config edit, not a code edit."""
    url = build([TEMPLATE], ['AAPL'])[0]
    params = parse_qs(urlsplit(url).query, keep_blank_values=True)
    assert params['cnt'] == ['100']
    assert params['page'] == ['1']
    assert params['xp'] == ['1'] and params['xs'] == ['1']
    assert urlsplit(url).netloc == 'openinsider.com'
    assert urlsplit(url).path == '/screener'


@pytest.mark.parametrize('tickers', [None, [], ['', '  ']])
def test_no_tickers_leaves_the_configured_url_untouched(tickers):
    assert build([TEMPLATE], tickers) == [TEMPLATE]


def test_a_template_without_the_parameter_still_gets_one():
    urls = build(['http://openinsider.com/screener?cnt=100'], ['AAPL'])
    assert _symbol(urls[0]) == 'AAPL'


def test_several_templates_each_expand():
    urls = build([TEMPLATE, TEMPLATE + '&x=2'], ['AAPL', 'MSFT'])
    assert len(urls) == 4
