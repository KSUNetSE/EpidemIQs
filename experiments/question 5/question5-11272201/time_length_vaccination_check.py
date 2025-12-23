
import pandas as pd
import os

filepaths = [os.path.join('output', f'results-{i}0.csv') for i in range(5)]

# Check unique times count and length of each file
times_info = {}
for fp in filepaths:
    df = pd.read_csv(fp)
    unique_times = df['time'].unique()
    times_info[fp] = (len(unique_times), len(df), unique_times.min(), unique_times.max())

# Also peek first 10 rows of each file
first_10_rows = {fp: pd.read_csv(fp).head(10) for fp in filepaths}
times_info, first_10_rows