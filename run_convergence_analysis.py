
"""
Run Convergence Analysis (v2)

This is the fourth and final module in the "Research Lab" pipeline. It analyzes
the output of the dual simulation to extract actionable insights and build a
"Strategy Playbook".

Methodology:
1.  **Load Dual Predictions:** Ingests the `dual_prediction_logs.parquet` file,
    which contains parallel predictions from the Main (technical) and Shadow
    (contextual) models.

2.  **Define Scenarios:** For each prediction, it categorizes the agreement between
    the two models:
    - `Convergence_Up`: Both models predict a positive outcome.
    - `Convergence_Down`: Both models predict a negative outcome.
    - `Divergence`: The models disagree on the direction.

3.  **Performance Analysis:** It groups the data by strategy (Ticker, Target) and
    scenario, then calculates performance metrics, most importantly the accuracy
    (`is_correct.mean()`).

4.  **Identify Alpha:** The analysis identifies scenarios where the main model's
    accuracy is significantly higher than its baseline performance. For example, it
    answers: "How much more accurate is the model when the shadow model agrees?"

Output:
-   `data/strategy_playbook.json`: A machine-readable file containing a list of
    high-performing strategies (plays), ready for the "Decision Agent".
-   `data/convergence_analysis_report.csv`: A human-readable report summarizing
    the performance of all tested scenarios.
"""

import logging
import pandas as pd
import numpy as np
import json

# --- Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Constants ---
PREDICTION_LOGS_PATH = "data/dual_prediction_logs.parquet"
PLAYBOOK_PATH = "data/strategy_playbook.json"
REPORT_PATH = "data.csv"

MIN_OBSERVATIONS_FOR_STATISTICS = 50
MIN_ACCURACY_IMPROVEMENT_FOR_PLAYBOOK = 0.05 # 5% improvement over baseline

def run_convergence_analysis():
    """Main function to run the convergence analysis."""
    logger.info("====== Starting Convergence Analysis Module (v2) ======")

    # 1. Load data
    try:
        df = pd.read_parquet(PREDICTION_LOGS_PATH)
        logger.info(f"Loaded dual prediction logs with shape {df.shape}")
    except FileNotFoundError:
        logger.error(f"{PREDICTION_LOGS_PATH} not found. Run the dual simulation first.")
        return

    # 2. Define Directions and Scenarios
    df.dropna(inplace=True)
    df['main_direction'] = np.sign(df['main_model_prediction'])
    df['shadow_direction'] = np.sign(df['shadow_model_prediction'])
    df['actual_direction'] = np.sign(df['actual_value'])
    df['is_correct'] = (df['main_direction'] == df['actual_direction']).astype(int)

    conditions = [
        (df['main_direction'] == 1) & (df['shadow_direction'] == 1),
        (df['main_direction'] == -1) & (df['shadow_direction'] == -1),
    ]
    choices = ['Convergence_Up', 'Convergence_Down']
    df['scenario'] = np.select(conditions, choices, default='Divergence')

    logger.info("Calculated scenarios for all predictions.")

    # 3. Group and Analyze
    analysis_results = []
    # Group by the core strategy components
    strategy_groups = df.groupby(['ticker', 'target'])

    for group_name, group_df in strategy_groups:
        ticker, target = group_name
        
        # Calculate baseline performance for the whole strategy
        baseline_accuracy = group_df['is_correct'].mean()
        baseline_observations = len(group_df)

        # Analyze performance for each scenario (Convergence, Divergence)
        scenario_groups = group_df.groupby('scenario')
        for scenario_name, scenario_df in scenario_groups:
            if len(scenario_df) < MIN_OBSERVATIONS_FOR_STATISTICS:
                continue

            scenario_accuracy = scenario_df['is_correct'].mean()
            accuracy_vs_baseline = scenario_accuracy - baseline_accuracy

            analysis_results.append({
                'ticker': ticker,
                'target': target,
                'scenario': scenario_name,
                'accuracy': round(scenario_accuracy, 4),
                'baseline_accuracy': round(baseline_accuracy, 4),
                'accuracy_improvement': round(accuracy_vs_baseline, 4),
                'observations': len(scenario_df)
            })

    if not analysis_results:
        logger.error("Analysis did not yield any results. Halting.")
        return

    # 4. Create and Save Report
    report_df = pd.DataFrame(analysis_results)
    report_df.sort_values(by=['ticker', 'target', 'accuracy_improvement'], ascending=False, inplace=True)
    report_df.to_csv(REPORT_PATH, index=False)
    logger.info(f"Convergence analysis report saved to {REPORT_PATH}")

    # 5. Generate and Save Strategy Playbook
    playbook = report_df[report_df['accuracy_improvement'] >= MIN_ACCURACY_IMPROVEMENT_FOR_PLAYBOOK].to_dict('records')
    
    with open(PLAYBOOK_PATH, 'w') as f:
        json.dump(playbook, f, indent=4)
    
    logger.info(f"Generated playbook with {len(playbook)} high-performing plays.")
    logger.info(f"Strategy Playbook saved to {PLAYBOOK_PATH}")
    logger.info("====== Convergence Analysis Module Finished ======")

if __name__ == "__main__":
    run_convergence_analysis()
