
import pandas as pd

file_path_42 = 'output/results-42.csv'
df42 = pd.read_csv(file_path_42)

header_42 = df42.columns.tolist()
shape_42 = df42.shape
head_42 = df42.head(5)

header_42, shape_42, head_42