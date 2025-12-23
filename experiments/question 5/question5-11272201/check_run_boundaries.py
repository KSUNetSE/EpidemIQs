
import pandas as pd

# Load first file fully
df = pd.read_csv('output/results-00.csv')

# Check first rows and last rows
head = df.head(10)
tail = df.tail(10)

# Check unique times count and if times reset
unique_times = df['time'].unique()
min_time = unique_times.min()
max_time = unique_times.max()

# Check if times repeat (for run boundaries) by checking if time goes back to 0
time_min_indices = df.index[df['time'] == 0.0].tolist()

head, tail, len(unique_times), min_time, max_time, len(time_min_indices), time_min_indices[:10]