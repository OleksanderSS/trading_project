from __future__ import annotations

import pandas as pd

from dean_os.replays.replay_evidence_window_selector import ReplayEvidenceWindowSelector


def _write_prices(path, start="2026-01-01", periods=80):
    dates = pd.date_range(start, periods=periods, freq="D", tz="UTC")
    rows = []
    for ticker, base in [("AMD", 100.0), ("NVDA", 200.0)]:
        for index, dt in enumerate(dates):
            rows.append({"datetime": dt.isoformat(), "ticker": ticker, "close": base + index})
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_replay_evidence_window_selector_finds_overlap_and_commands(tmp_path):
    prices = _write_prices(tmp_path / "prices.csv")
    news = tmp_path / "news.csv"
    pd.DataFrame(
        [
            {
                "published_date": "2026-02-01T00:00:00Z",
                "title": "AMD and NVDA demand update",
                "content": "Data center momentum for AMD and NVDA.",
            }
        ]
    ).to_csv(news, index=False)

    payload = ReplayEvidenceWindowSelector(tmp_path / "reports").build(
        price_data_path=prices,
        news_data_paths=[news],
        tickers=["AMD", "NVDA"],
        lookback_days=10,
        horizon_days=[10],
        step_days=5,
        start_as_of="2026-02-05T00:00:00+00:00",
        end_as_of="2026-02-15T00:00:00+00:00",
        save=False,
    )

    assert payload["summary"]["selection_status"] == "windows_ready"
    assert payload["summary"]["eligible_window_count"] == 2
    assert payload["eligible_windows"][0]["evidence_rows"] == 1
    assert payload["eligible_windows"][0]["ticker_hits"]["AMD"] >= 1
    assert "--lookback-days 10" in payload["commands"]["historical_research_replay_batch"]
    assert str(news) in payload["commands"]["historical_research_replay_batch"]
    assert payload["safety"]["pipeline_run_performed"] is False


def test_replay_evidence_window_selector_blocks_when_future_prices_are_too_short(tmp_path):
    prices = _write_prices(tmp_path / "prices.csv", start="2026-01-01", periods=45)
    news = tmp_path / "news.csv"
    pd.DataFrame([{"published_date": "2026-02-01T00:00:00Z", "title": "AMD update"}]).to_csv(news, index=False)

    payload = ReplayEvidenceWindowSelector(tmp_path / "reports").build(
        price_data_path=prices,
        news_data_paths=[news],
        tickers=["AMD"],
        lookback_days=10,
        horizon_days=[30],
        step_days=1,
        start_as_of="2026-02-05T00:00:00+00:00",
        end_as_of="2026-02-05T00:00:00+00:00",
        save=False,
    )

    assert payload["summary"]["selection_status"] == "no_eligible_windows"
    assert payload["summary"]["eligible_window_count"] == 0
    assert payload["rejected_windows_sample"][0]["blockers"] == ["missing_price_tickers"]
    assert payload["commands"]["historical_research_replay_batch"] is None


def test_replay_evidence_window_selector_reports_no_sources(tmp_path):
    prices = _write_prices(tmp_path / "prices.csv")

    payload = ReplayEvidenceWindowSelector(tmp_path / "reports").build(
        price_data_path=prices,
        tickers=["AMD"],
        lookback_days=10,
        horizon_days=[10],
        start_as_of="2026-02-05T00:00:00+00:00",
        end_as_of="2026-02-05T00:00:00+00:00",
        save=False,
    )

    assert payload["summary"]["selection_status"] == "no_eligible_windows"
    assert payload["rejected_windows_sample"][0]["blockers"] == ["not_enough_evidence_rows", "not_enough_sources"]
    assert payload["summary"]["loaded_source_count"] == 0
