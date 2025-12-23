
import pandas as pd
import numpy as np

# Load the available results files and combine them into one DataFrame with run identifier
files = ['output/results-11.csv', 'output/results-12.csv', 'output/results-21.csv', 'output/results-22.csv']

# Load each file and add a run id
all_runs = []
for idx, f in enumerate(files, start=1):
    df = pd.read_csv(f)
    df['run'] = idx
    all_runs.append(df)

combined_data = pd.concat(all_runs, ignore_index=True)

# Check combined data shape and basic info
combined_shape = combined_data.shape
combined_columns = combined_data.columns.tolist()
combined_shape, combined_columns