
# Check peak infection times and their I values for each file
import pandas as pd
import os

filepaths = [os.path.join('output', f'results-{i}0.csv') for i in range(5)]
peak_time_info = {}

for fp in filepaths:
    df = pd.read_csv(fp)
    idx_max_I = df['I'].idxmax()
    peak_time = df.loc[idx_max_I, 'time']
    peak_I = df.loc[idx_max_I, 'I']
    # Also check times and I around the peak (one before and after if exists)
    peak_data = df.loc[max(0, idx_max_I-1): idx_max_I+1, ['time', 'I']]
    peak_time_info[os.path.basename(fp)] = {'peak_time': peak_time, 'peak_I': peak_I, 'peak_data': peak_data}

peak_time_info