
import pandas as pd

# Load the data to take a look at its structure
file_path = 'output/results-01.csv'
data = pd.read_csv(file_path)
data.head(), data.columns, data.shape