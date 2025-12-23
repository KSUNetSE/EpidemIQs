
import pandas as pd

# Load the data
file_path = '/Users/hosseinsamaei/phd/epidemiqs/output/results-12.csv'
data = pd.read_csv(file_path)

# Inspect the first few rows and general info of the dataset to understand its structure
head = data.head()
info = data.info()
columns = data.columns.tolist()

head, columns