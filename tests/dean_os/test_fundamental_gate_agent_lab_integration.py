from __future__ import annotations

import asyncio
import json

from dean_os.agent_lab import AgentLabRunner
from dean_os.agents.domain_research import ValueScreeningAgent
from dean_os.fundamental_input_readiness_gate import FundamentalInputReadinessGate
from dean_os.schemas import MarketContext


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _value_fundamentals():
    return {
        "AMD": {
            "pe": 10.0,
            "pb": 1.0,
            "debt_to_equity": 0.5,
            "fcf_yield": 0.08,
            "roe": 0.15,
        }
    }


def _structured_value_fundamentals():
    return {
        "AMD": {
            "metrics": {
                metric_name: {
                    "value": value,
                    "unit": (
                        "ratio"
                        if metric_name
                        in {
                            "pe",
                            "pb",
                            "debt_to_equity",
                            "fcf_yield",
                            "roe",
                        }
                        else "USD"
                    ),
                    "period": "FY2025",
                }
                for metric_name, value in _value_fundamentals()[
                    "AMD"
                ].items()
            },
            "available_at": "2026-06-29T16:00:00+00:00",
            "source_url": (
                "https://example.test/filings/amd-fy2025"
            ),
        }
    }


def _gate_metadata(payload):
    summary = payload["summary"]
    guidance = payload["decision_guidance"]
    return {
        "gate_attached": True,
        "run_id": payload["run_id"],
        "readiness_status": summary["readiness_status"],
        "can_enter_manual_fundamental_review": summary["can_enter_manual_fundamental_review"],
        "can_feed_value_screening_after_manual_review": summary["can_feed_value_screening_after_manual_review"],
        "metric_count": summary["metric_count"],
        "structured_accepted_fingerprint": summary[
            "structured_accepted_fingerprint"
        ],
        "warning_count": guidance["warning_count"],
        "fail_count": guidance["fail_count"],
    }


def _clean_gate(tmp_path):
    rows = []
    for metric_name, value in _value_fundamentals()["AMD"].items():
        rows.append(
            {
                "ticker": "AMD",
                "metric_name": metric_name,
                "value": value,
                "unit": "ratio",
                "period": "FY2025",
                "available_at": "2026-06-29T16:00:00+00:00",
                "source": (
                    "https://example.test/filings/amd-fy2025"
                ),
            }
        )
    source_path = _write_json(tmp_path / "clean_fundamentals.json", {"extracted_fundamental_metrics": rows})
    return FundamentalInputReadinessGate(tmp_path / "gate").build(
        fundamentals_json=source_path,
        as_of="2026-06-30T12:00:00+00:00",
        save=False,
    )


def _warning_gate(tmp_path):
    source_path = _write_json(tmp_path / "warning_fundamentals.json", {"fundamentals": _value_fundamentals()})
    return FundamentalInputReadinessGate(tmp_path / "gate").build(
        fundamentals_json=source_path,
        as_of="2026-06-30T12:00:00+00:00",
        save=False,
    )


def test_value_screening_blocks_when_attached_fundamental_gate_has_warnings(tmp_path):
    gate = _warning_gate(tmp_path)
    context = MarketContext(
        as_of="2026-06-30T12:00:00+00:00",
        tickers=["AMD"],
        fundamentals=_structured_value_fundamentals(),
        metadata={"fundamental_input_readiness_gate": _gate_metadata(gate)},
    )

    report = asyncio.run(ValueScreeningAgent(name="value_screening", config={}).run(context))

    assert report.verdict == "needs_more_data"
    assert report.position_bias == "insufficient_data"
    assert report.valuation_gap is None
    assert "Fundamental values were not used for scoring." in report.blind_spots


def test_value_screening_allows_clean_attached_fundamental_gate(tmp_path):
    gate = _clean_gate(tmp_path)
    context = MarketContext(
        as_of="2026-06-30T12:00:00+00:00",
        tickers=["AMD"],
        fundamentals=_structured_value_fundamentals(),
        metadata={"fundamental_input_readiness_gate": _gate_metadata(gate)},
    )

    report = asyncio.run(ValueScreeningAgent(name="value_screening", config={}).run(context))

    assert report.verdict == "undervalued"
    assert report.ticker == "AMD"
    assert report.valuation_gap == "best_value_score=1.00; average_value_score=1.00"
    assert any("FundamentalInputReadinessGate is attached and clean" in risk for risk in report.risks)


def test_agent_lab_records_fundamental_gate_summary(tmp_path):
    gate = _warning_gate(tmp_path)

    report = asyncio.run(
        AgentLabRunner(
            corpus_path=tmp_path / "corpus.sqlite",
            learning_path=tmp_path / "learning.sqlite",
            output_dir=tmp_path / "agent_lab",
            memory_path=tmp_path / "memory.sqlite",
            log_path=None,
        ).run(
            documents=[],
            tickers=["AMD"],
            fundamentals=_structured_value_fundamentals(),
            fundamental_gate=gate,
            as_of="2026-06-30T12:00:00+00:00",
            include_financial_nlp=False,
            include_synthesis=False,
            include_operations_proposals=False,
            create_learning_records=False,
        )
    )

    summary = report.summary["fundamental_input_readiness_gate"]
    assert summary["gate_attached"] is True
    assert summary["readiness_status"] == (
        "fundamental_input_structured_contract_blocked"
    )
    assert summary["can_feed_value_screening_after_manual_review"] is False
    value_report = next(item for item in report.reports if item.agent_name == "value_screening")
    assert value_report.verdict == "needs_more_data"
