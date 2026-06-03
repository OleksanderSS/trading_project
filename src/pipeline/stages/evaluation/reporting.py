import logging
from pathlib import Path
from typing import Any, Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def plot_equity_curve(portfolio_history, financial_metrics=None) -> Path:
    out = Path("reports/charts")
    out.mkdir(parents=True, exist_ok=True)
    p = out / "equity_curve.png"

    if hasattr(portfolio_history, "get") and isinstance(portfolio_history, dict):
        dates = portfolio_history.get("dates")
        values = portfolio_history.get("values")
    elif hasattr(portfolio_history, "index") and "total_value" in getattr(portfolio_history, "columns", []):
        dates = portfolio_history.index
        values = portfolio_history["total_value"]
    else:
        dates, values = None, None

    try:
        if dates is not None and values is not None:
            plt.figure(figsize=(12, 6))
            plt.plot(dates, values, color="green", linewidth=2)
            title = "Equity Curve"
            if isinstance(financial_metrics, dict):
                title += f" | Return: {financial_metrics.get('total_return_pct', 0):.2%}"
            plt.title(title)
            plt.grid(True, alpha=0.3)
            plt.savefig(p)
            plt.close()
    except Exception as e:
        logger.error(f"Failed to plot equity curve: {e}", exc_info=True)
        raise
    return p


def save_report(path: Path, summary: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        import json

        json.dump(summary, f, indent=2)
        f.write("\n")
