
import pandas as pd

file_path_31 = 'output/results-31.csv'
df31 = pd.read_csv(file_path_31)

header_31 = df31.columns.tolist()
shape_31 = df31.shape
head_31 = df31.head(5)

header_31, shape_31, head_31