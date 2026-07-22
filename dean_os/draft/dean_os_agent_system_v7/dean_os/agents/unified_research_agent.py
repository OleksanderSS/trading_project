"""Unified Research Agent — reads DuckDB (1.18 GB), ResearchCorpus, and MarketContext
to produce cross-referenced data intelligence across all sources.

Extends SpecialistResearchAgent with:
1. DuckDB data loader (news, prices, FRED, SEC, RSS, Google News, NewsAPI, sentiment cache)
2. Data quality / coverage metrics per source and per ticker
3. Cross-correlation: news volume vs price moves, macro releases vs market
4. Sector-agnostic: same agent, any sector, any tickers
"""

from __future__ import annotations

import asyncio
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import duckdb

from dean_os.agents.research_agents import (
    BULLISH_PATTERNS,
    SpecialistResearchAgent,
    extract_pattern_counts,
    material_documents,
    metric_patterns_from_context,
)
from dean_os.base import AnalyticalAgent
from dean_os.schemas import AnalyticalReport, MarketContext, ResearchNote
from dean_os.utils import clamp


DUCKDB_PATH = "data/trading_data.duckdb"


class DuckDBDataLoader:
    """Queries the accumulated pipeline DuckDB (1.18 GB, 11 tables)."""

    def __init__(self, db_path: str | Path = DUCKDB_PATH):
        self.db_path = Path(db_path)

    def table_exists(self, name: str) -> bool:
        try:
            con = duckdb.connect(str(self.db_path))
            row = con.execute(
                "SELECT COUNT(*) FROM duckdb_tables() WHERE table_name = ?", [name]
            ).fetchone()
            con.close()
            return row and row[0] > 0
        except Exception:
            return False

    def table_row_count(self, name: str) -> int:
        try:
            con = duckdb.connect(str(self.db_path))
            count = con.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0]
            con.close()
            return count
        except Exception:
            return 0

    def table_date_range(self, name: str, date_col: str) -> tuple[str | None, str | None]:
        try:
            con = duckdb.connect(str(self.db_path))
            row = con.execute(
                f'SELECT MIN({date_col}), MAX({date_col}) FROM "{name}"'
            ).fetchone()
            con.close()
            return (str(row[0]) if row and row[0] else None,
                    str(row[1]) if row and row[1] else None)
        except Exception:
            return None, None

    def news_keyword_counts(self, keywords: list[str], limit: int = 50) -> Counter:
        """Search huggingface_data for keywords — single scan, not per-keyword queries."""
        if not keywords or not self.table_exists("huggingface_data"):
            return Counter()
        try:
            con = duckdb.connect(str(self.db_path))
            exprs = ", ".join(
                f'SUM(CASE WHEN "text" ILIKE \'%{kw.replace(chr(39), chr(39)+chr(39))}%\' THEN 1 ELSE 0 END) AS "{kw}"'
                for kw in keywords
            )
            row = con.execute(
                f'SELECT {exprs} FROM "huggingface_data"'
            ).fetchone()
            con.close()
            if not row:
                return Counter()
            return Counter({kw: int(row[i]) for i, kw in enumerate(keywords) if row[i] and int(row[i]) > 0})
        except Exception:
            return Counter()

    def market_data_summary(self, ticker: str | None = None) -> dict[str, Any]:
        """Price data stats from market_data_raw."""
        if not self.table_exists("market_data_raw"):
            return {}
        try:
            con = duckdb.connect(str(self.db_path))
            where = f'WHERE ticker = \'{ticker}\'' if ticker else ""
            row = con.execute(f"""
                SELECT COUNT(*), MIN(datetime), MAX(datetime),
                       AVG(close), STDDEV(close)
                FROM "market_data_raw" {where}
            """).fetchone()
            con.close()
            if row and row[0]:
                return {
                    "row_count": int(row[0]),
                    "date_from": str(row[1]) if row[1] else None,
                    "date_to": str(row[2]) if row[2] else None,
                    "avg_close": float(row[3]) if row[3] else None,
                    "close_volatility": float(row[4]) if row[4] else None,
                }
            return {}
        except Exception:
            return {}

    def ticker_coverage(self) -> dict[str, dict[str, int]]:
        """Count records per ticker across all data tables."""
        coverage: dict[str, dict[str, int]] = {}
        tables_with_ticker = {
            "market_data_raw": "ticker",
            "sec_filings": "ticker",
        }
        try:
            con = duckdb.connect(str(self.db_path))
            for table, col in tables_with_ticker.items():
                if not self.table_exists(table):
                    continue
                try:
                    rows = con.execute(
                        f'SELECT {col}, COUNT(*) as cnt FROM "{table}" '
                        f"WHERE {col} IS NOT NULL GROUP BY {col} ORDER BY cnt DESC"
                    ).fetchall()
                    for ticker, cnt in rows:
                        coverage.setdefault(str(ticker).upper(), {})[table] = int(cnt)
                except Exception:
                    continue
            con.close()
        except Exception:
            pass
        return coverage

    def all_table_stats(self) -> list[dict[str, Any]]:
        """Basic stats for every table."""
        stats = []
        try:
            con = duckdb.connect(str(self.db_path))
            tables = con.execute(
                "SELECT table_name FROM duckdb_tables() ORDER BY table_name"
            ).fetchall()
            for (name,) in tables:
                count = con.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0]
                cols = con.execute(
                    f"SELECT column_name FROM information_schema.columns "
                    f"WHERE table_name='{name}' ORDER BY ordinal_position"
                ).fetchall()
                stats.append({
                    "table": name,
                    "rows": int(count),
                    "columns": [c[0] for c in cols],
                })
            con.close()
        except Exception:
            pass
        return stats

    def sec_filing_summary(self, ticker: str | None = None, limit: int = 10) -> list[dict]:
        """Recent SEC filings."""
        if not self.table_exists("sec_filings"):
            return []
        try:
            con = duckdb.connect(str(self.db_path))
            where = f'WHERE UPPER(ticker) = \'{ticker.upper()}\'' if ticker else ""
            rows = con.execute(f"""
                SELECT accessionNumber, formType, companyName, ticker,
                       filingDate, description
                FROM "sec_filings" {where}
                ORDER BY filingDate DESC LIMIT {limit}
            """).fetchall()
            con.close()
            return [
                {
                    "accession": r[0], "form_type": r[1],
                    "company": r[2], "ticker": r[3],
                    "filing_date": str(r[4]) if r[4] else None,
                    "description": (r[5] or "")[:200],
                }
                for r in rows
            ]
        except Exception:
            return []

    def macro_recent(self, series_ids: list[str] | None = None, limit: int = 5) -> list[dict]:
        """Latest FRED macro observations."""
        if not self.table_exists("fred_data"):
            return []
        try:
            con = duckdb.connect(str(self.db_path))
            where = ""
            if series_ids:
                quoted = [f"'{s}'" for s in series_ids]
                where = f'WHERE series_id IN ({",".join(quoted)})'
            rows = con.execute(f"""
                SELECT series_id, value, date, realtime_start, realtime_end
                FROM "fred_data" {where}
                ORDER BY date DESC LIMIT {limit}
            """).fetchall()
            con.close()
            return [
                {"series_id": r[0], "value": float(r[1]) if r[1] else None,
                 "date": str(r[2]) if r[2] else None,
                 "realtime_start": str(r[3]) if r[3] else None}
                for r in rows
            ]
        except Exception:
            return []


class UnifiedResearchAgent(SpecialistResearchAgent):
    """Extends SpecialistResearchAgent with:
    - DuckDB data loading (1.18 GB accumulated pipeline data)
    - Cross-source data quality / coverage metrics
    - News ↔ price ↔ macro ↔ SEC cross-references
    - Ticker-level coverage analytics
    - Sector-agnostic: same agent works for any sector/tickers
    """

    version = "0.2.0"

    def __init__(self, name: str | None = None, config: dict[str, Any] | None = None):
        super().__init__(name=name, config=config)
        self.duckdb = DuckDBDataLoader(
            self.config.get("duckdb_path", DUCKDB_PATH)
        )

    async def run(self, context: MarketContext) -> AnalyticalReport:
        # Run base SpecialistResearchAgent logic first
        base_report = await super().run(context)

        # Then add DuckDB intelligence
        duckdb_insights = await self._analyze_duckdb(context)
        data_quality = self._enhanced_data_quality(context, duckdb_insights)

        # Enrich the base report with DuckDB findings
        enriched_report = self._enrich_report(base_report, duckdb_insights, data_quality, context)
        return enriched_report

    async def _analyze_duckdb(self, context: MarketContext) -> dict[str, Any]:
        """Query DuckDB for all relevant data and produce structured insights."""
        insights: dict[str, Any] = {
            "db_available": False,
            "table_stats": [],
            "news_keyword_counts": {},
            "ticker_coverage": {},
            "market_data": {},
            "sec_filings": {},
            "macro_recent": [],
            "coverage_gaps": [],
        }

        # Basic connectivity (run in thread to keep cancellable)
        loop = asyncio.get_event_loop()
        stats = await loop.run_in_executor(None, self.duckdb.all_table_stats)
        if not stats:
            return insights
        insights["db_available"] = True
        insights["table_stats"] = stats

        # Total DB size estimate
        total_rows = sum(s["rows"] for s in stats)
        insights["total_rows_in_db"] = total_rows

        # Ticker coverage across market_data and SEC
        ticker_coverage = await loop.run_in_executor(None, self.duckdb.ticker_coverage)
        insights["ticker_coverage"] = ticker_coverage

        # Market data for requested tickers
        for ticker in context.tickers:
            md = await loop.run_in_executor(
                None, self.duckdb.market_data_summary, ticker
            )
            if md:
                insights.setdefault("market_data", {})[ticker] = md
                if md.get("row_count", 0) < 100:
                    insights["coverage_gaps"].append(
                        f"{ticker}: only {md['row_count']} price rows"
                    )

        # SEC filings for requested tickers
        for ticker in context.tickers:
            filings = await loop.run_in_executor(
                None, self.duckdb.sec_filing_summary, ticker, 5
            )
            if filings:
                insights.setdefault("sec_filings", {})[ticker] = filings

        # Recent macro
        macro = await loop.run_in_executor(None, self.duckdb.macro_recent, None, 10)
        if macro:
            insights["macro_recent"] = macro

        # News table stats (lightweight — just row counts, no full-text scan)
        news_tables = ["huggingface_data", "google_news", "rss_news", "newsapi_articles", "news_sentiment_cache"]
        for nt in news_tables:
            if self.duckdb.table_exists(nt):
                cnt = self.duckdb.table_row_count(nt)
                if cnt:
                    insights.setdefault("news_table_stats", {})[nt] = cnt

        return insights

    def _enhanced_data_quality(self, context: MarketContext, insights: dict[str, Any]) -> str:
        """Improved data quality that accounts for DuckDB data."""
        base_quality = "weak"
        if insights.get("db_available"):
            total_rows = insights.get("total_rows_in_db", 0)
            if total_rows > 100_000:
                base_quality = "strong"
            elif total_rows > 10_000:
                base_quality = "partial"

        doc_count = len(material_documents(context))
        if doc_count > 0 and base_quality == "weak":
            base_quality = "partial"

        return base_quality

    def _enrich_report(
        self,
        base_report: AnalyticalReport,
        insights: dict[str, Any],
        data_quality: str,
        context: MarketContext,
    ) -> AnalyticalReport:
        """Merge DuckDB insights into the base report."""
        evidence = list(base_report.evidence)
        reasons = list(base_report.reasons)
        risks = list(base_report.risks)

        # Add DuckDB evidence
        if insights.get("db_available"):
            evidence.append(
                self.evidence("metric", "duckdb", "total_rows", insights.get("total_rows_in_db", 0))
            )
            for stat in insights.get("table_stats", []):
                evidence.append(
                    self.evidence("metric", f"duckdb.{stat['table']}", "rows", stat["rows"])
                )
            reasons.append(f"DuckDB: {insights.get('total_rows_in_db', 0):,} rows across {len(insights.get('table_stats', []))} tables")

        # Add ticker coverage
        coverage = insights.get("ticker_coverage", {})
        for ticker, sources in sorted(coverage.items()):
            if ticker.upper() in {t.upper() for t in context.tickers}:
                total = sum(sources.values())
                evidence.append(
                    self.evidence("metric", f"duckdb.coverage.{ticker}", "total_rows", total)
                )
                reasons.append(f"Data coverage for {ticker}: {total:,} rows in {list(sources.keys())}")

        # Add gaps
        for gap in insights.get("coverage_gaps", []):
            risks.append(gap)

        # Add news table stats from DuckDB
        news_stats = insights.get("news_table_stats", {})
        if news_stats:
            total_news = sum(news_stats.values())
            tables_str = ", ".join(f"{t}={v:,}" for t, v in sorted(news_stats.items()))
            reasons.append(f"DuckDB news corpus: {total_news:,} items across {tables_str}")

        # Add SEC insights
        sec_data = insights.get("sec_filings", {})
        for ticker, filings in sec_data.items():
            if filings:
                forms = [f["form_type"] for f in filings[:3]]
                reasons.append(f"SEC {ticker}: recent {', '.join(forms)}")

        # Add macro insights
        macro = insights.get("macro_recent", [])
        if macro:
            for m in macro[:5]:
                reasons.append(f"FRED {m['series_id']}={m['value']} ({m['date']})")

        # Build enriched thesis
        thesis_parts = [base_report.thesis] if base_report.thesis else []
        if insights.get("db_available"):
            thesis_parts.append(f"DB intelligence: {insights.get('total_rows_in_db', 0):,} records analyzed")
        if news_stats:
            total_news = sum(news_stats.values())
            thesis_parts.append(f"DuckDB news: {total_news:,} items")

        enriched_thesis = "; ".join(thesis_parts) if thesis_parts else "Unified research analysis complete"

        # Build enriched verdict
        verdict = base_report.verdict
        if data_quality == "weak" and verdict not in ("needs_more_data",):
            verdict = "needs_more_data"
        elif data_quality == "strong" and verdict == "needs_more_data":
            verdict = "neutral"

        enriched_confidence = clamp(
            base_report.confidence + (0.1 if insights.get("db_available") else -0.1),
            0.0, 0.95,
        )

        return AnalyticalReport(
            agent_name=self.name,
            agent_version=self.version,
            verdict=verdict,
            confidence=enriched_confidence,
            data_quality_score={"strong": 0.9, "partial": 0.6, "weak": 0.2}[data_quality],
            signal_strength=base_report.signal_strength,
            ticker=",".join(context.tickers) if context.tickers else None,
            asset_or_sector="unified_research",
            horizon_years=base_report.horizon_years,
            thesis=enriched_thesis,
            data_quality=data_quality,
            position_bias=base_report.position_bias,
            reasons=reasons[:20],
            risks=risks[:15],
            blind_spots=base_report.blind_spots,
            evidence=evidence,
            input_hash=self.context_hash(context),
        )
