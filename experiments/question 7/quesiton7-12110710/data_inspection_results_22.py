
import pandas as pd

file_path_22 = 'output/results-22.csv'
df22 = pd.read_csv(file_path_22)

header_22 = df22.columns.tolist()
shape_22 = df22.shape
head_22 = df22.head(5)

header_22, shape_22, head_22