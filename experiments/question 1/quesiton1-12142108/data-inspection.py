
import pandas as pd

# Load the data from the provided CSV path
data_path = '/Users/hosseinsamaei/phd/epidemiqs/output/results-11.csv'
data = pd.read_csv(data_path)

# Inspect the first few rows and columns to understand the data structure

first_rows = data.head()
columns = data.columns
shape = data.shape

first_rows, columns, shape