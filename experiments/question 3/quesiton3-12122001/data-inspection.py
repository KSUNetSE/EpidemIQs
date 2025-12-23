
import pandas as pd

# Load the data to inspect its structure
file_path = 'output/results-12.csv'
data = pd.read_csv(file_path)

# Display basic info about the data
info = data.info(memory_usage='deep')

# Display the first few rows to understand the data layout
head = data.head()

# Extract columns and shape
columns = data.columns.tolist()
shape = data.shape

info, head, columns, shape