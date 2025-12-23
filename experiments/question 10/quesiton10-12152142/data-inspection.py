
import pandas as pd

# Load the data to inspect its structure
file_path = '/Users/hosseinsamaei/phd/epidemiqs/output/results-21-summary.csv'
data = pd.read_csv(file_path)  

# View the first few rows and the columns to understand the data
columns = data.columns.tolist()
data_head = data.head()

data_shape = data.shape

columns, data_head, data_shape