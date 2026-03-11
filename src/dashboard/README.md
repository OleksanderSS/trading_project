# 📈 Interactive UI & Visual Monitoring Center

This directory contains the code for the main `Streamlit`-based user interface of the trading system. It serves as the **Interactive UI & Visual Monitoring Center**, providing a comprehensive, real-time overview of the system's performance, active signals, market news, and overall health.

---

## 🚀 `main_app.py` - The Core of the UI

The `UnifiedDashboard` class in `main_app.py` is the central Streamlit/Dash application designed for visualizing portfolio performance, model forecasts, and market context. It acts as the 'Human-in-the-loop' gateway of the system.

### Key Features:

1.  **Unified Data Source:** The dashboard is fully integrated with the project's data architecture. It reads real-time data directly via **`src/data/management/data_manager.py`** and displays pre-calculated performance reports generated during **Stage 7 (Evaluation)**.

2.  **Consistent Visualization:** To ensure "one version of the truth," the dashboard utilizes **`src/analytics/reporting/visualization.py`** for all charts and equity curves. This ensures that the visuals seen on the screen are identical to those in the automated PDF/Markdown reports.

3.  **Dynamic Configuration:** The UI is not hard-coded. It dynamically populates filters and components based on the central `unified_config.yaml` and asset manifests, ensuring the interface is always in sync with the active trading universe.

4.  **Comprehensive Tabs:**
    *   **`[DATA] Overview`**: High-level system metrics and aggregated performance.
    *   **`[UP] Trading Signals`**: Latest signals with confidence levels and model attribution.
    *   **`News Analysis`**: Market sentiment and news feed analysis.
    *   **`[WARN] Risk Management`**: Detailed model breakdowns (Sharpe, Drawdown) and portfolio exposure.
    *   **`System Monitoring`**: Real-time hardware health (CPU/RAM) and database integrity stats.

### ⚙️ Backend Dependencies

For the dashboard to be fully functional, the system must populate the following schemas via the **DataManager**:

*   **`trading_signals`**: Stores all generated signals and their model-specific confidence.
*   **`model_performance`**: Contains validated metrics for each trained model.
*   **`news`**: Aggregated market intelligence from various collectors.
*   **`evaluation_summary`**: Post-inference analysis results from Stage 7.

---

### How to Run

To start the dashboard, navigate to the project's root directory and run:

```bash
streamlit run src/dashboard/main_app.py
```