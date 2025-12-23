
import pandas as pd

# Load the data
file_path = 'output/results-11.csv'
data = pd.read_csv(file_path)

# Check the shape and columns of the data to understand its structure
shape = data.shape
columns = data.columns.tolist()

# Display the first few rows to understand the contents
head = data.head()

shape, columns, head