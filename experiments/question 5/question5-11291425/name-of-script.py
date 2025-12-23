
import pandas as pd

# Load the data to inspect its structure
file_path = 'output/results-61.csv'
data = pd.read_csv(file_path)

# Inspect the first few rows and columns info to understand the structure
head = data.head()
info = data.info()
cols = data.columns.tolist()
num_rows = len(data)