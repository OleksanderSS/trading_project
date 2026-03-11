# Monitoring Module (`src/monitoring`)

This module is responsible for **Operational Health & Model Performance Tracking**, ensuring the trading system remains robust and its predictive "edge" is maintained after deployment.

## Components

### 1. System Health Monitor (`system_health_monitor.py`)
Provides real-time tracking of infrastructure health and resource utilization.
*   **RAM & CPU:** Monitors memory consumption and processor load to prevent leaks and bottlenecks during large-scale processing (Stage 0/7).
*   **Database Health:** Ensures connectivity and integrity of the DuckDB storage and external API integrations.
*   **Alerting Hooks:** Triggers critical system alerts if hardware thresholds are exceeded.

### 2. ML Analytics (`ml_analytics.py`)
Focuses on the statistical performance and reliability of deployed Machine Learning models.
*   **Drift Detection:** Identifies when model inputs or target distributions change significantly, signaling a need for retraining.
*   **Accuracy Degradation:** Continuously monitors hit rates and directional consistency using data from the **Experience Diary**.
*   **Feature Importance Monitoring:** Tracks the ongoing relevance of features to ensure models aren't relying on stale data patterns.

### 3. Comprehensive Reporter (`comprehensive_reporter.py`)
Acts as a central aggregator for the monitoring suite.
*   **Data Aggregation:** Combines hardware telemetry and software performance metrics into a unified health state.
*   **Automated Alerts:** Generates structured health reports and emergency notifications for the system operator.

## Integration & Purpose
*   **Alerting:** Fully integrated with **src/core/logging/notifier.py** for cross-platform alerting (Telegram, Email, etc.).
*   **Visualization:** Monitoring data and health trends are visualized in **src/analytics/reporting/** and evaluated during **Stage 7 (Evaluation)**.
*   **System Integrity:** The suite ensures the system remains performant and mathematically sound, preventing silent failures due to changing market conditions or infrastructure issues.