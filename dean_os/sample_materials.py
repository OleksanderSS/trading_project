from __future__ import annotations

from dean_os.schemas import ResearchDocument


def agent_lab_sample_documents(
    tickers: list[str] | None = None,
    sectors: list[str] | None = None,
    tags: list[str] | None = None,
) -> list[ResearchDocument]:
    """Small deterministic corpus for testing the agent loop without external data."""

    tickers = tickers or ["AMD", "NVDA"]
    sectors = sectors or ["semiconductor"]
    tags = [*(tags or []), "sample", "agent_lab_smoke"]
    return [
        ResearchDocument(
            title="Sample AI Compute Cycle Brief",
            source_type="article",
            text=(
                "AI data center accelerator demand, semiconductor capital expenditure, GPU supply, "
                "and cloud compute investment can create a multi-year compute capital cycle. "
                "Backlog growth, advanced packaging constraints, and enterprise AI adoption may support "
                "revenue growth for leading semiconductor companies, while export controls and capacity "
                "shortages remain material risks."
            ),
            tickers=tickers,
            sectors=sectors,
            tags=tags,
            metadata={"sample": True},
        ),
        ResearchDocument(
            title="Sample Value Discipline Notes",
            source_type="book",
            text=(
                "A margin of safety comes from free cash flow, book value discipline, balance sheet strength, "
                "and buying good businesses at a discount. Companies with pricing power, durable returns on "
                "capital, and conservative leverage can compound when earnings revisions improve."
            ),
            tickers=tickers,
            sectors=sectors,
            tags=tags,
            metadata={"sample": True},
        ),
    ]
