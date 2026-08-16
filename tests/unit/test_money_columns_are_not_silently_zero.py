"""A number stored the way a page displayed it is still a number.

`insider_trades` keeps all three of its numeric columns as text, exactly as
scraped:

    value     '-$4,962,488'   '+$10,681,309'
    price     '$522.37'
    quantity  '-9,500'        '+637,200'

`pd.to_numeric` returns NaN for every one of them -- 1,395 of 1,395 rows -- and
NaN sums to zero, so `insider_net_value_30d` read as a confident 0.0 on all
three timeframes rather than as missing. The enricher had a fallback to
`price * quantity`, and it failed identically, both halves being the same kind
of string.

This is the `''` is not NaN failure wearing different clothes: the column
exists, it is populated, every presence check passes, and the value is a
plausible-looking zero.
"""

import numpy as np
import pandas as pd
import pytest

from src.features.enrichers.base import BaseEnricher

parse = BaseEnricher.parse_money


@pytest.mark.parametrize(
    'text, expected',
    [
        ('-$4,962,488', -4962488.0),
        ('+$10,681,309', 10681309.0),
        ('$522.37', 522.37),
        ('-9,500', -9500.0),
        ('+637,200', 637200.0),
        ('73820', 73820.0),
        ('  $1,000.50  ', 1000.50),
        ('(2,500)', -2500.0),      # accountant's negative
        ('€1.234', 1.234),
        ('', np.nan),
        ('n/a', np.nan),
    ],
)
def test_a_formatted_number_is_read_as_its_value(text, expected):
    got = parse(pd.Series([text])).iloc[0]
    if pd.isna(expected):
        assert pd.isna(got)
    else:
        assert got == pytest.approx(expected)


def test_the_whole_column_survives_not_just_the_tidy_rows():
    column = pd.Series(['-$4,962,488', '+$10,681,309', '$522.37', '(2,500)'])
    assert parse(column).notna().all()


def test_plain_numbers_are_untouched():
    column = pd.Series([1.5, -2, 0])
    assert list(parse(column)) == [1.5, -2.0, 0.0]


def test_a_missing_value_stays_missing_rather_than_becoming_zero():
    """The distinction the defect erased: nothing filed is not zero filed."""
    parsed = parse(pd.Series(['', None, 'n/a']))
    assert parsed.isna().all()
    assert parsed.sum() == 0.0, 'sum of NaN is 0 — which is why this must stay NaN'


def test_the_insider_feature_is_no_longer_structurally_zero():
    from src.features.enrichers.ticker_external_enricher import TickerExternalEnricher

    filings = pd.DataFrame({
        'ticker': ['AAPL', 'AAPL', 'MSFT'],
        'filing_date': pd.to_datetime(['2026-07-02', '2026-07-10', '2026-07-05']),
        'trade_date': pd.to_datetime(['2026-07-01', '2026-07-09', '2026-07-04']),
        'trade_type': ['P - Purchase', 'S - Sale', 'P - Purchase'],
        'price': ['$100.00', '$110.00', '$50.00'],
        'quantity': ['+1,000', '-2,000', '+3,000'],
        'value': ['+$100,000', '-$220,000', '+$150,000'],
    })
    days = pd.date_range('2026-07-01', periods=30, freq='B')
    bars = pd.DataFrame({
        'datetime': list(days) * 2,
        'ticker': ['AAPL'] * len(days) + ['MSFT'] * len(days),
        'close': 1.0,
    })

    enriched = TickerExternalEnricher().enrich(bars, insider_trades=filings)
    net = enriched['insider_net_value_30d'].fillna(0.0)
    assert net.nunique() > 1, 'the feature is still one repeated value'
    assert (net != 0).any(), 'no bar carries a filed value'
    assert (net < 0).any(), 'a sale must be able to make the net negative'


def test_a_sale_is_not_counted_as_a_purchase():
    """The defect the parsing fix exposed.

    The source signs its own figures -- '-$220,000' for a sale -- and the
    enricher negated sales on top of that, so a disposal ADDED to the net.
    Invisible while every figure was NaN.
    """
    from src.features.enrichers.ticker_external_enricher import TickerExternalEnricher

    filings = pd.DataFrame({
        'ticker': ['AAPL'],
        'filing_date': pd.to_datetime(['2026-07-02']),
        'trade_date': pd.to_datetime(['2026-07-01']),
        'trade_type': ['S - Sale'],
        'price': ['$110.00'],
        'quantity': ['-2,000'],
        'value': ['-$220,000'],
    })
    days = pd.date_range('2026-07-01', periods=10, freq='B')
    bars = pd.DataFrame({'datetime': days, 'ticker': 'AAPL', 'close': 1.0})

    net = TickerExternalEnricher().enrich(bars, insider_trades=filings)
    net = net['insider_net_value_30d'].dropna()
    assert (net <= 0).all(), 'a sale increased the net insider value'
    assert net.min() == pytest.approx(-220000.0)


def test_an_unsigned_source_still_gets_its_sign_from_the_trade_type():
    """Not every feed signs its figures; the trade type must still work."""
    from src.features.enrichers.ticker_external_enricher import TickerExternalEnricher

    filings = pd.DataFrame({
        'ticker': ['AAPL'],
        'filing_date': pd.to_datetime(['2026-07-02']),
        'trade_date': pd.to_datetime(['2026-07-01']),
        'trade_type': ['S - Sale'],
        'price': ['$110.00'],
        'quantity': ['2,000'],
        'value': ['$220,000'],
    })
    days = pd.date_range('2026-07-01', periods=10, freq='B')
    bars = pd.DataFrame({'datetime': days, 'ticker': 'AAPL', 'close': 1.0})

    net = TickerExternalEnricher().enrich(bars, insider_trades=filings)
    net = net['insider_net_value_30d'].dropna()
    assert net.min() == pytest.approx(-220000.0)
