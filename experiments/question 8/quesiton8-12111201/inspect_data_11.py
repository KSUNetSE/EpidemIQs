
import pandas as pd

# Load one sample file to inspect the data structure
file_path = 'output/results-11.csv'
df = pd.read_csv(file_path)

# Extract basic info about the dataset
columns = df.columns.tolist()
head = df.head()
shape = df.shape