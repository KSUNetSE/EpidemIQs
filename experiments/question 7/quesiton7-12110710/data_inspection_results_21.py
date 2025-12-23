
import pandas as pd

file_path_21 = 'output/results-21.csv'
df21 = pd.read_csv(file_path_21)

header_21 = df21.columns.tolist()
shape_21 = df21.shape
head_21 = df21.head(5)

header_21, shape_21, head_21