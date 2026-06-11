import pandas as pd

# Load the classified findings
df = pd.read_csv('diagnostic_reports/risk_findings_classified.csv')

# Filter for review required
review_df = df[df['status'] == 'REVIEW_REQUIRED']

# Show distribution by file path
distribution = review_df['file'].value_counts()

print("Distribution of REVIEW_REQUIRED findings by file:")
print(distribution.head(20))
