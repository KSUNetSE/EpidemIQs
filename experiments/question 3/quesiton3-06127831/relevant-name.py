
import pandas as pd

# Load the data and check its structure
file_path = 'output/results-12.csv'
data = pd.read_csv(file_path)
data.head(), data.columns, data.shape