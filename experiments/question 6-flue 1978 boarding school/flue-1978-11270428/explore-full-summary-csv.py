
# Explore the entire data structure for one file to examine all columns and understand the data fully
file_path = 'output/results-11-summary.csv'
df = pd.read_csv(file_path)

# Get all columns and info
columns = df.columns.tolist()
description = df.describe(include='all')

columns, description