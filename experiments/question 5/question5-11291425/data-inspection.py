
import pandas as pd

# Load the data
file_path = 'output/results-31.csv'
data = pd.read_csv(file_path)

# Inspect the data to understand its structure
info = data.info()
head = data.head()

info, head
import pandas as pd

# Load the data
file_path = 'output/results-51.csv'
data = pd.read_csv(file_path)

# Display basic info and first few rows to understand the structure
info = data.info()
head = data.head()
columns = data.columns.tolist()
data_shape = data.shape

info, head, columns, data_shape