
import pandas as pd

# Load the data files
files = {
    'results_11': '/Users/hosseinsamaei/phd/epidemiqs/output/results-11.csv',
    'results_12': '/Users/hosseinsamaei/phd/epidemiqs/output/results-12.csv',
    'results_13': '/Users/hosseinsamaei/phd/epidemiqs/output/results-13.csv',
    'results_14': '/Users/hosseinsamaei/phd/epidemiqs/output/results-14.csv',
    'results_16': '/Users/hosseinsamaei/phd/epidemiqs/output/results-16.csv'
}

# Read the csv files into pandas dataframes
results_11 = pd.read_csv(files['results_11'])
results_12 = pd.read_csv(files['results_12'])
results_13 = pd.read_csv(files['results_13'])
results_14 = pd.read_csv(files['results_14'])
results_16 = pd.read_csv(files['results_16'])

# Check the column names and first few rows to understand the structure
summary = {}
for key, df in zip(files.keys(), [results_11, results_12, results_13, results_14, results_16]):
    summary[key] = {
        'columns': list(df.columns),
        'head': df.head().to_dict(orient='records'),
        'shape': df.shape
    }

summary