
import pandas as pd

# Load the data from all three files
results_10 = pd.read_csv('output/results-10.csv')
results_11 = pd.read_csv('output/results-11.csv')
results_summary = pd.read_csv('output/results-summary.csv')

# Inspecting the first few rows and column names of each data set
info_10 = (results_10.shape, results_10.columns.tolist(), results_10.head())
info_11 = (results_11.shape, results_11.columns.tolist(), results_11.head())
info_summary = (results_summary.shape, results_summary.columns.tolist(), results_summary.head())