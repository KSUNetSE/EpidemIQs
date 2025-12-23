
import pandas as pd

# File paths
file_paths = [
    'output/results-11-summary.csv',
    'output/results-12-summary.csv',
    'output/results-13-summary.csv',
    'output/results-14-summary.csv'
]

# Read the first few rows of each file to understand the structure
samples = {}
for path in file_paths:
    df = pd.read_csv(path)
    samples[path] = df.head()

samples