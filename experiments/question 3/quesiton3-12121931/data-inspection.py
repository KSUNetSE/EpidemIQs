
import pandas as pd

# Load the data to inspect its structure
file_path = 'output/results-11.csv'
data = pd.read_csv(file_path)

# Inspect the first few rows and columns to understand the structure
columns = data.columns
data_sample = data.head()
data_shape = data.shape
import pandas as pd

# Load the data to get an overview of its structure
file_path = 'output/results-12.csv'
data = pd.read_csv(file_path)

# Inspect first few rows and columns to understand structure
head = data.head()
columns = data.columns
shape = data.shape

head, columns, shape