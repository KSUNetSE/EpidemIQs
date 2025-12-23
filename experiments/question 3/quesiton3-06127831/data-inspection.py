
import pandas as pd

# Load the data
file_path = 'output/results-13.csv'
data = pd.read_csv(file_path)

# Display first few rows and columns info to understand the structure of the data
head = data.head()
columns = data.columns
head, list(columns)
import pandas as pd

# Load the dataset to examine its structure
file_path = 'output/results-16.csv'
data = pd.read_csv(file_path)

# Display the first few rows and columns information to understand its structure
head = data.head()
columns = data.columns.tolist()
head, columns
import pandas as pd

# Load the data from the specified file path
file_path = 'output/results-14.csv'
data = pd.read_csv(file_path)
# Check the first few rows and columns to understand the structure and contents of the data
head_data = data.head()
columns = data.columns.tolist()
head_data, columns