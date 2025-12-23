
import pandas as pd

file_path_61 = 'output/results-61.csv'
df61 = pd.read_csv(file_path_61)

header_61 = df61.columns.tolist()
shape_61 = df61.shape
head_61 = df61.head(5)

(header_61, shape_61, head_61)