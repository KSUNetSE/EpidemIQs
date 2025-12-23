
import pandas as pd

file_path_52 = 'output/results-52.csv'
df52 = pd.read_csv(file_path_52)

header_52 = df52.columns.tolist()
shape_52 = df52.shape
head_52 = df52.head(5)

header_52, shape_52, head_52