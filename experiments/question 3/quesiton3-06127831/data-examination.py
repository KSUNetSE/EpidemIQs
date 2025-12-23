
import pandas as pd

# Load the data from the specified CSV file
file_path = 'output/results-15.csv'
data = pd.read_csv(file_path)

# Display the first few rows to understand the structure and columns of the data
head_data = data.head()
columns = data.columns.tolist()

head_data, columns