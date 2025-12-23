
import pandas as pd

data_path = 'output/results-21.csv'

# Load the data
data = pd.read_csv(data_path)

# Inspect first 5 rows
data_head = data.head()

# Get columns
data_columns = data.columns.tolist()

# Get data shape
data_shape = data.shape