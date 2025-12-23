
import pandas as pd

# Load the data
file_path = '/Users/hosseinsamaei/phd/epidemiqs/output/results-10.csv'
data = pd.read_csv(file_path)

# Inspect the first few rows and columns to understand the structure
sample_data = data.head(), data.columns, data.shape

import pandas as pd

# Load the data
file_path = '/Users/hosseinsamaei/phd/epidemiqs/output/results-11.csv'
data = pd.read_csv(file_path)

# Inspect the first few rows and data structure
info = data.info()
head = data.head()
cols = data.columns.tolist()
import pandas as pd

# Load the data
file_path = '/Users/hosseinsamaei/phd/epidemiqs/output/results-12.csv'
data = pd.read_csv(file_path)

# Check the basic information about the data
info = data.info()

# Preview the first few rows to understand structure
head = data.head()

info, head
import pandas as pd

# Load the data to examine the contents and structure
file_path = '/Users/hosseinsamaei/phd/epidemiqs/output/results-13.csv'
data = pd.read_csv(file_path)

# Get an overview of the data: columns, first few rows, and data size
columns = data.columns.tolist()
head_data = data.head()
shape = data.shape

columns, head_data, shape