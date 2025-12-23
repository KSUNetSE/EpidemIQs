
import pandas as pd

# Load all the files to get an initial feel on their sizes
file_paths = ['output/results-11.csv', 'output/results-21.csv', 'output/results-31.csv', 'output/results-41.csv']
shapes = {file: pd.read_csv(file).shape for file in file_paths}