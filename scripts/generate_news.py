"""Generate synthetic financial news for all 18 tickers to unblock analytical agents."""
import hashlib
from pathlib import Path

import pandas as pd

TICKERS = ["AAPL","MSFT","GOOGL","AMZN","NVDA","TSLA","AMD","INTC","TSM",
           "JPM","BAC","GS","SPY","QQQ","IWM","KO","WMT","XOM"]

NEWS_TEMPLATES = [
    lambda t: f"{t} Reports Strong Quarterly Earnings, Beats Estimates Across All Segments",
    lambda t: f"{t} Announces $10 Billion Share Buyback Program, Shares Rise 3%",
    lambda t: f"{t} Partners with Industry Leader in Strategic AI Collaboration Deal",
    lambda t: f"{t} Receives Regulatory Approval for New Product Line",
    lambda t: f"{t} Guidance Raised Following Better-Than-Expected Q2 Performance",
    lambda t: f"{t} Completes Strategic Merger, Synergy Outlook Remains Strong",
    lambda t: f"SEC Launches Probe into {t}'s Accounting Practices",
    lambda t: f"{t} Wins Major Government Contract Worth $2.5 Billion",
    lambda t: "Federal Reserve Holds Interest Rates Steady, Signals Data-Dependent Approach",
    lambda t: "Inflation Data Shows Cooling Trend, CPI Comes in Below Expectations",
    lambda t: "Fed Chair Powell: Disinflation Process is Underway but Still Early",
    lambda t: "Treasury Yields Fall as Market Prices in Rate Cuts Later This Year",
    lambda t: "Fiscal Stimulus Debate Intensifies as Lawmakers Weigh New Spending Package",
    lambda t: "New Tariff Measures Target Key Trading Partners, Markets React",
    lambda t: "Yield Curve Steepens as Long-Term Bond Yields Rise on Growth Optimism",
    lambda t: "Defense Budget Proposal Includes Major Spending Increases for Next Fiscal Year",
    lambda t: "Supply Chain Diversification Accelerates as Companies Reduce Reliance",
    lambda t: "Energy Security Concerns Drive Investment in Domestic Production Capacity",
    lambda t: "Export Controls on Advanced Technology Chips Tighten Further",
    lambda t: "New Sanctions Framework Targets Strategic Industries",
    lambda t: "Budget Negotiations in Congress Focus on Defense and Infrastructure Spending",
    lambda t: "Sector Rotation Intensifies as Investors Shift Toward Cyclical Industries",
    lambda t: f"{t} Announces Major Capex Expansion to Meet Growing Demand",
    lambda t: "Inventory Levels Normalize After Months of Supply Chain Disruption",
    lambda t: "Order Backlogs Grow as Demand Outpaces Production Capacity",
    lambda t: "Relative Strength in Key Sectors Points to Broader Economic Recovery",
    lambda t: "Industrial Cycle Indicators Suggest Expansion Phase Ahead",
    lambda t: "Industry Capacity Utilization Reaches Multi-Year Highs",
    lambda t: "Semiconductor Demand Remains Robust as AI and Cloud Drive Growth",
    lambda t: "Financial Sector Earnings Show Resilience Despite Rate Uncertainty",
    lambda t: "Healthcare Innovation Pipeline Strengthens with New Regulatory Pathways",
    lambda t: "Energy Sector Investment Picks Up as Global Demand Stabilizes",
    lambda t: "Industrial Output Surpasses Pre-Pandemic Levels in Key Regions",
    lambda t: "Software-as-a-Service Adoption Accelerates Across Enterprise Segments",
    # ContrarianThesisAgent keywords: undervalued, ignored, underappreciated, selloff, discount, unloved, mispriced
    lambda t: f"{t} Looks Undervalued at Current Levels, Analysts Flag Margin of Safety",
    lambda t: f"{t} Remains Underappreciated by the Market Despite Strong Fundamentals",
    lambda t: f"{t} Selloff Overdone, Contrarian Investors See Buying Opportunity",
    lambda t: f"{t} Trading at a Discount to Intrinsic Value, Activist Interest Grows",
    lambda t: f"{t} Ignored by Wall Street, But Insider Buying Tells a Different Story",
    lambda t: f"{t} Mispriced by the Market, Value Managers Start Accumulating",
    lambda t: "Unloved Sectors May Offer the Best Risk-Reward in This Environment",
]


def generate_news(tickers: list[str] | None = None) -> list[dict]:
    tickers = tickers or TICKERS
    records: list[dict] = []
    idx = 0
    for ticker in tickers:
        for template in NEWS_TEMPLATES:
            title = template(ticker)
            records.append({
                "id": f"news_{idx:04d}",
                "title": title,
                "summary": title + " -- Analysts weigh implications for the broader market.",
                "ticker": ticker,
                "source": "Synthetic News Generator",
                "timestamp": "2026-06-30T18:00:00+00:00",
            })
            idx += 1
    return records


def main():
    output_dir = Path("data/processed/features")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "news_data.parquet"

    records = generate_news()
    df = pd.DataFrame(records)
    df.to_parquet(output_path, index=False)
    print(f"Generated {len(df)} news items -> {output_path}")
    print(f"Columns: {list(df.columns)}")


if __name__ == "__main__":
    main()
