
import pandas as pd

file_path_41 = 'output/results-41.csv'
df41 = pd.read_csv(file_path_41)

header_41 = df41.columns.tolist()
shape_41 = df41.shape
head_41 = df41.head(5)

header_41, shape_41, head_41