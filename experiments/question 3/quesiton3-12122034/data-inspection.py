
import pandas as pd

# Load the data
file_path = 'output/results-11.csv'
data = pd.read_csv(file_path)

# Inspect the structure of the data
info = data.info()
head = data.head()
cols = data.columns
data_shape = data.shape

info, head, cols, data_shape
import pandas as pd

# Load the data
file_path = 'output/results-22.csv'
data = pd.read_csv(file_path)

# Inspect the first few rows and the columns of the data
head_data = data.head()
columns = data.columns.tolist()
shape = data.shape
head_data, columns, shape
import pandas as pd

# Load the data to inspect its structure
file_path = 'output/results-21.csv'
data = pd.read_csv(file_path)

# Inspect the first few rows and the columns of the data
data_head = data.head()
data_columns = data.columns.tolist()
data_shape = data.shape
