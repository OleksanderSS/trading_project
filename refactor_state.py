import os
import re


def refactor_mutable_attributes(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    new_lines = []
    class_name = None
    
    # Simple regex to find class definition
    class_def_pattern = re.compile(r'class\s+(\w+)')
    # Regex to find mutable assignments like: ATTRIBUTE = []
    mutable_pattern = re.compile(r'^\s+(\w+)\s*=\s*(\[\]|\{\})')
    
    in_class = False
    attributes_to_move = []
    
    for line in lines:
        class_match = class_def_pattern.search(line)
        if class_match:
            class_name = class_match.group(1)
            in_class = True
            new_lines.append(line)
            continue
            
        if in_class:
            mut_match = mutable_pattern.search(line)
            if mut_match:
                attr_name = mut_match.group(1)
                attr_val = mut_match.group(2)
                attributes_to_move.append((attr_name, attr_val))
                continue # Remove from class body
        
        # Insert __init__ if not exists and we have attributes to move
        if in_class and attributes_to_move and 'def __init__(self' in line:
            new_lines.append(line)
            for attr, val in attributes_to_move:
                new_lines.append(f"        self.{attr} = {val}\n")
            attributes_to_move = [] # Cleared
            continue

        new_lines.append(line)

    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)

# List of files identified by the audit report as having mutable class attributes
files_to_fix = [
    'src/pipeline/guards/temporal_leakage_guard.py',
    'src/pipeline/guards/temporal_target_guard.py',
    'src/features/analysis/news_decay_modeler.py',
    'src/features/analysis/regime_importance_tracker.py',
    'src/models/analysis/regime_winner_analyzer.py',
    'src/models/ensemble/model_correlation_analyzer.py',
    'src/models/monitoring/prediction_drift_monitor.py',
    'src/analytics/analyzers/model_comparison_analyzer.py',
    'src/analytics/context/market_phase_analyzer.py',
    'src/analytics/data_managers/model_results_manager.py',
    'src/data/management/handlers/connection_handler.py',
    'src/features/validation/redundancy_detector.py',
    'src/models/ensemble/confidence_calibrator.py',
    'src/models/ensemble/weight_stability_monitor.py',
    'src/pipeline/guards/macro_release_timing_guard.py',
    'src/pipeline/guards/timeframe_alignment_guard.py',
    'src/risk/kill_switch_manager.py'
]

for file in files_to_fix:
    if os.path.exists(file):
        print(f"Refactoring {file}...")
        refactor_mutable_attributes(file)
