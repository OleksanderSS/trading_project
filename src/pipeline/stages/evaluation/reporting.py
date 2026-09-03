import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def plot_equity_curve(portfolio_history, financial_metrics=None,
                      simulated: bool = False) -> Path:
    """Draw the equity curve, and say on the image when it is not real.

    When Stage 5 produces too few signals, `backtest_analyzer` substitutes
    randomly-generated data and sets `is_simulated_data`. That flag reaches
    `summary_*.json`, so a PROGRAM can tell. Nothing reached the picture: on
    2026-08-30 the run wrote `equity_curve.png` from random numbers, finished
    with `Pipeline completed successfully`, and exited zero. A person opening
    that file saw an equity curve and had no way to know.

    Every other defect found in this pipeline handed back emptiness. This one
    hands back something that looks like a result, which is why the marking
    goes on the image itself rather than only in the log.
    """
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
            title = "SIMULATED DATA — NOT A RESULT" if simulated else "Equity Curve"
            if isinstance(financial_metrics, dict):
                title += f" | Return: {financial_metrics.get('total_return_pct', 0):.2%}"
            plt.title(title, color="red" if simulated else "black",
                      fontweight="bold" if simulated else "normal")
            if simulated:
                plt.gcf().text(
                    0.5, 0.5,
                    "RANDOMLY GENERATED / signals too thin for a backtest",
                    fontsize=26, color="red", alpha=0.35,
                    ha="center", va="center", rotation=20,
                )
            plt.grid(True, alpha=0.3)
            plt.savefig(p)
            plt.close()
    except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
        logger.exception(f"Failed to plot equity curve: {e}")
        raise
    return p


def save_report(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        import json

        json.dump(summary, f, indent=2, default=str)
        f.write("\n")
