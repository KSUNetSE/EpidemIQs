
import pandas as pd

# Load the CSV data
file_path = 'output/results-11.csv'
df = pd.read_csv(file_path)

# Inspect the first few rows and columns
header = df.columns.tolist()
shape = df.shape
head = df.head(5)

header, shape, head