
import pandas as pd

# List of CSV file paths
file_paths = [
    'output/results-00.csv', 'output/results-01.csv', 'output/results-02.csv', 'output/results-03.csv', 'output/results-04.csv',
    'output/results-05.csv', 'output/results-06.csv', 'output/results-07.csv', 'output/results-08.csv', 'output/results-09.csv'
]

# Read first file to inspect the structure
df = pd.read_csv(file_paths[0])
df_head = df.head()
df_columns = df.columns
df_shape = df.shape

# Output for inspection
result = {
    'columns': list(df_columns),
    'head': df_head.to_dict(),
    'shape': df_shape
}