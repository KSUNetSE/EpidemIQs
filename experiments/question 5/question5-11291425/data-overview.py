
import pandas as pd

# Load data
file_path = 'output/results-41.csv'
data = pd.read_csv(file_path)

# Check the basic structure of the data
num_rows, num_cols = data.shape
columns = data.columns.tolist()
data_head = data.head()

num_rows, num_cols, columns, data_head