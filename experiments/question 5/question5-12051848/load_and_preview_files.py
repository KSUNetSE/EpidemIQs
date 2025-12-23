
import pandas as pd

# Load all CSVs initially to inspect their structure and headers
filepaths = ['output/results-11.csv', 'output/results-12.csv', 'output/results-13.csv', 'output/results-14.csv', 'output/results-15.csv', 'output/results-16.csv', 'output/results-1attackrates.csv', 'output/results-21.csv', 'output/results-31.csv', 'output/results-41.csv', 'output/results-42.csv', 'output/results-43.csv', 'output/results-44.csv', 'output/results-45.csv', 'output/results-46.csv', 'output/results-4attackrates.csv']

# Read first few rows of each file to inspect
files_preview = {}
for fp in filepaths:
    df = pd.read_csv(fp)
    files_preview[fp] = {
        'columns': df.columns.tolist(),
        'head': df.head(3),
        'shape': df.shape
    }

files_preview