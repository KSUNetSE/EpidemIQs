
import pandas as pd

# Load the data from the provided path
file_path = '/Users/hosseinsamaei/phd/epidemiqs/output/results-11-summary.csv'
data = pd.read_csv(file_path)

# Inspect the basic structure: columns and the first few rows
columns = data.columns.tolist()
sample_data = data.head()
data_shape = data.shape