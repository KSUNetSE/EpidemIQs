
import pandas as pd

# Load the two datasets
file_path_temporal = 'output/results-11.csv'
file_path_static = 'output/results-12.csv'

# Load the csv files into pandas dataframes
results_temporal = pd.read_csv(file_path_temporal)
results_static = pd.read_csv(file_path_static)

# Get the basic information about the datasets
info_temporal = results_temporal.info(), results_temporal.head()
info_static = results_static.info(), results_static.head()
