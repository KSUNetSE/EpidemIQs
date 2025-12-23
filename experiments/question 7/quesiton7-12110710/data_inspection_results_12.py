
import pandas as pd

# Load the CSV data
file_path = 'output/results-12.csv'
df2 = pd.read_csv(file_path)

# Inspect columns, shape, and first few rows
header2 = df2.columns.tolist()
shape2 = df2.shape
head2 = df2.head(5)

header2, shape2, head2