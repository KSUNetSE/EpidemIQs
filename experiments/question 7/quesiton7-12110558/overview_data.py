
import pandas as pd

# Load all the CSV files
file_paths = [
    'output/results-11.csv',  # core seeded, kc=0.90
    'output/results-21.csv',  # periphery seeded, kc=0.90
    'output/results-14.csv',  # core seeded, kc=1.00
    'output/results-24.csv'   # periphery seeded, kc=1.00
]

dataframes = {path: pd.read_csv(path) for path in file_paths}

# Extract columns overview to understand what we have
columns_overview = {path: df.columns.tolist() for path, df in dataframes.items()}

# Also check the size of each dataframe
sizes = {path: df.shape for path, df in dataframes.items()}

columns_overview, sizes