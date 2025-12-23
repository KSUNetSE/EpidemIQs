
import pandas as pd

# File paths list
file_paths = [
    'output/results-21.csv',
    'output/results-22.csv',
    'output/results-23.csv',
    'output/results-24.csv',
    'output/results-25.csv',
    'output/results-26.csv'
]

# Read first few rows of each file to inspect structure
sample_dfs = {fp: pd.read_csv(fp).head() for fp in file_paths}

# Get columns and shape info
data_info = {fp: {'columns': df.columns.tolist(), 'shape': df.shape} for fp, df in sample_dfs.items()}

data_info