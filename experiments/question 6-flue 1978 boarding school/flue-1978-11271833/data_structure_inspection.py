
import pandas as pd
import os
# Load all metrics CSV files
files = ['output/results-11-metrics.csv', 'output/results-12-metrics.csv', 'output/results-13-metrics.csv', 'output/results-14-metrics.csv', 'output/results-15-metrics.csv', 'output/results-16-metrics.csv']
data_dict = {file: pd.read_csv(file) for file in files}

# Extract shape and head of each to understand structure
summary = {file: {'shape': df.shape, 'columns': df.columns.tolist(), 'head': df.head().to_dict()} for file, df in data_dict.items()}
summary