
import pandas as pd

# Load the CSV file to examine its structure
file_path = '/Users/hosseinsamaei/phd/epidemiqs/output/results-12-metrics.csv'
data = pd.read_csv(file_path)

# Check the first few rows and columns (structure) to understand what metrics are available
head_data = data.head()
columns = data.columns.tolist()

head_data, columns
import pandas as pd

# Load the data from the provided file path
file_path = '/Users/hosseinsamaei/phd/epidemiqs/output/results-11-metrics.csv'
data = pd.read_csv(file_path)

# Check the first few rows and columns to understand the structure and columns of the dataset
sample_data = data.head()
columns = data.columns.tolist()

sample_data, columns