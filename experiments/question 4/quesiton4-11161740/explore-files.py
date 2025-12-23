
import pandas as pd
import numpy as np

# File paths
file_paths = [
    'output/results-11.csv',
    'output/results-12.csv',
    'output/results-13.csv',
    'output/results-14.csv',
    'output/results-15.csv',
    'output/results-16.csv'
]

# Load all files into a dictionary of DataFrames
all_data = {fp: pd.read_csv(fp) for fp in file_paths}

# Check the general structure (columns and first few rows) of each file
file_summaries = {}
for fp, df in all_data.items():
    file_summaries[fp] = {
        'columns': df.columns.tolist(),
        'head': df.head().to_dict(orient='records'),
        'shape': df.shape
    }

file_summaries