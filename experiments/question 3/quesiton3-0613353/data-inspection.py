
import pandas as pd

# Load the data for examination
file_path = 'output/results-11.csv'
data = pd.read_csv(file_path)
data.head(), data.columns, data.shape