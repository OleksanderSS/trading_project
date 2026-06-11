import pandas as pd

# Load the audit findings
df = pd.read_csv('diagnostic_reports/risk_findings.csv')

# Define safe directories and files
safe_dirs = ['cli/', 'devtools/', 'monitoring/', 'training/', 'pipeline/hybrid/', 'pipeline/stages/', 'pipeline/guards/']
safe_files = ['training/adaptive_training_manager.py', 'features/feature_selection_cache.py', 'features/enrichers/derived_features_enricher.py', 'meta_learning/memory/knn_context_finder.py', 'validation/temporal_feature_separator.py', 'analytics/calculators/econometrics_calculator.py', 'models/model_selector/smart_selector.py']

# Function to mark as safe
def classify_finding(row):
    if any(row['file'].startswith(d) for d in safe_dirs) or row['file'] in safe_files:
        return 'SAFE_INFRASTRUCTURE'
    return 'REVIEW_REQUIRED'

# Apply classification
df['status'] = df.apply(classify_finding, axis=1)

# Output summary
print("Classification Summary:")
print(df['status'].value_counts())

# Save the updated findings to a temporary file for verification
df.to_csv('diagnostic_reports/risk_findings_classified.csv', index=False)
print("\nClassified findings saved to diagnostic_reports/risk_findings_classified.csv")
