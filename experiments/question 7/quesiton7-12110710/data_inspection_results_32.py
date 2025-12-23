
import pandas as pd

file_path32 = 'output/results-32.csv'
df32 = pd.read_csv(file_path32)

header32 = df32.columns.tolist()
shape32 = df32.shape
head32 = df32.head(5)

header32, shape32, head32