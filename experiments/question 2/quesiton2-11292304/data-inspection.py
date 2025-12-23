
import pandas as pd

# Load the data
file_path = '/Users/hosseinsamaei/phd/epidemiqs/output/results-10.csv'
data = pd.read_csv(file_path)

# Quick look at first few rows and the columns
head = data.head()
columns = data.columns.tolist()
shape = data.shape

import pandas as pd
# Load the data to understand the structure
file_path = '/Users/hosseinsamaei/phd/epidemiqs/output/results-11.csv'
data = pd.read_csv(file_path)

# Inspect basic information about the data
info = data.info()
head = data.head()
shape = data.shape
import pandas as pd

# Load the data
file_path = '/Users/hosseinsamaei/phd/epidemiqs/output/results-12.csv'
data = pd.read_csv(file_path)

# Inspect the structure of the dataframe
rows, cols = data.shape
columns = data.columns.tolist()

rows, cols, columns
import pandas as pd

# Load the data for inspection
file_path = '/Users/hosseinsamaei/phd/epidemiqs/output/results-14.csv'
data = pd.read_csv(file_path)

# Get basic info about the data
shape = data.shape
head = data.head()
columns = data.columns.tolist()

shape, columns, head
import pandas as pd

# Load the data
file_path = '/Users/hosseinsamaei/phd/epidemiqs/output/results-13.csv'
data = pd.read_csv(file_path)

# Get basic info about data structure
info = data.info()
# Display first few rows to understand columns
head = data.head()
# Get summary statistics
describe = data.describe()

info, head, describe