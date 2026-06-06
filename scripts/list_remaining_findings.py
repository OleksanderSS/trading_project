import pandas as pd

# Load the classified findings
df = pd.read_csv('diagnostic_reports/risk_findings_classified.csv')

# Filter for review required
review_df = df[df['status'] == 'REVIEW_REQUIRED']

# Show distribution by file path
distribution = review_df['file'].value_counts()

print("Top 10 files with most remaining REVIEW_REQUIRED findings:")
print(distribution.head(10))
