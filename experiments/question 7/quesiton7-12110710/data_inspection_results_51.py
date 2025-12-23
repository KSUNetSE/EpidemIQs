
import pandas as pd

file_path_51 = 'output/results-51.csv'
df51 = pd.read_csv(file_path_51)

header_51 = df51.columns.tolist()
shape_51 = df51.shape
head_51 = df51.head(5)

(header_51, shape_51, head_51)