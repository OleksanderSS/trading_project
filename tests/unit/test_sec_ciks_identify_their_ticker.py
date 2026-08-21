"""A CIK that resolves is not a CIK that is right.

SPY was configured as 0000896976. That number returns HTTP 200 -- for 'VAN
KAMPEN AMERICAN CAPITAL EQUITY OPPORTUNITY TRUST SER 14', an unrelated entity
whose 24 filings all fall between 1995 and 2001. None of them landed inside
the collection window, so the collector raised nothing and SPY contributed
zero filings on every run. The one that DID error, IWM, was the visible one
and the less harmful one.

Audited against SEC on 2026-08-21. These tests hit no network: they pin the
configuration and the collector's handling, so a silent regression cannot
reintroduce a number nobody checked.
"""

import io
import re

import pytest
import yaml

_KNOWN_GOOD = {
    "SPY": 884394,      # SPDR S&P 500 ETF TRUST
    "QQQ": 1067839,     # INVESCO QQQ TRUST, SERIES 1
    "XOM": 34088,       # EXXON MOBIL CORP
    "AAPL": 320193,
    "MSFT": 789019,
    "TSLA": 1318605,
    "NVDA": 1045810,
}

#: Numbers measured to be wrong, so a revert is a test failure rather than a
#: silent return to collecting somebody else's filings.
_KNOWN_WRONG = {
    "SPY": 896976,      # a dead Van Kampen trust, last filed 2001
    "IWM": 1112953,     # 404; IWM is not a filer in SEC's own ticker map
}


@pytest.fixture(scope="module")
def details():
    cfg = yaml.safe_load(io.open("src/config/assets.yaml", encoding="utf-8"))

    def _find(node, key):
        if isinstance(node, dict):
            if key in node:
                return node[key]
            for value in node.values():
                found = _find(value, key)
                if found is not None:
                    return found
        return None

    found = _find(cfg, "details")
    assert found is not None, "assets.yaml has no details block"
    return found


@pytest.mark.parametrize("ticker,cik", sorted(_KNOWN_GOOD.items()))
def test_the_audited_ciks_are_the_configured_ones(details, ticker, cik):
    assert details.get(ticker, {}).get("cik") == cik


@pytest.mark.parametrize("ticker,wrong", sorted(_KNOWN_WRONG.items()))
def test_a_number_measured_to_be_wrong_does_not_come_back(details, ticker, wrong):
    assert details.get(ticker, {}).get("cik") != wrong


def test_iwm_carries_no_cik_at_all(details):
    """A wrong number that errors every run is worse than no number."""
    assert "cik" not in details.get("IWM", {})


def test_every_configured_cik_is_a_positive_integer(details):
    for ticker, data in details.items():
        if not isinstance(data, dict) or "cik" not in data:
            continue
        cik = data["cik"]
        assert isinstance(cik, int), f"{ticker}: {cik!r} is not an int"
        assert 0 < cik < 10 ** 10, f"{ticker}: {cik} is not a plausible CIK"


def test_no_two_tickers_share_a_cik(details):
    """One filer, one ticker here: a shared number means one of them is wrong."""
    seen: dict[int, str] = {}
    for ticker, data in details.items():
        if not isinstance(data, dict) or "cik" not in data:
            continue
        cik = data["cik"]
        assert cik not in seen, f"{ticker} and {seen[cik]} both claim CIK {cik}"
        seen[cik] = ticker


def test_the_collector_names_the_entity_it_fetched():
    """So the next mismatch shows up in the log, not in an audit months later."""
    source = io.open(
        "src/data/collectors/sec_filings_collector.py", encoding="utf-8"
    ).read()
    assert re.search(r'logger\.info\(\s*"\[SEC\] %s -> CIK%s %r"', source), (
        "the collector must log the entity name SEC returned"
    )


def test_a_missing_submissions_file_is_a_warning_not_a_stack_trace():
    source = io.open(
        "src/data/collectors/sec_filings_collector.py", encoding="utf-8"
    ).read()
    assert "No submissions for %s at CIK%s" in source
    assert "company_tickers.json" in source, "the warning must say where to look"
