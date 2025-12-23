
import pandas as pd

file_path_62 = 'output/results-62.csv'
df62 = pd.read_csv(file_path_62)

header_62 = df62.columns.tolist()
shape_62 = df62.shape
head_62 = df62.head(5)

(header_62, shape_62, head_62)