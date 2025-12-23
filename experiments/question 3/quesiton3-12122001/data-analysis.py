
import pandas as pd

# Load the data
file_path = 'output/results-13.csv'
data = pd.read_csv(file_path)

# Check first few rows to understand the structure
head_data = data.head()

# Check column names
data_columns = data.columns.tolist()

# Check data shape
shape = data.shape

head_data, data_columns, shape