
import numpy as np

# Maximum time in dataset
max_time = data['time'].max()

# Extract data at maximum time for final epidemic size
final_data = data[data['time'] == max_time][['R', 'I', 'E', 'S']].iloc[0]

# Find max infectious and time
peak_idx = data['I'].idxmax()
peak_time = data.loc[peak_idx, 'time']
peak_I = data.loc[peak_idx, 'I']

# Find first non-zero infectious time and last time I>=1 (epidemic duration)
nonzero_I = data[data['I'] > 0]
epidemic_start_time = nonzero_I['time'].min()
epidemic_end_time = nonzero_I[nonzero_I['I'] >= 1]['time'].max()
epidemic_duration = epidemic_end_time - epidemic_start_time

# Initial seeds from t=0
initial_seeds = data[data['time'] == 0][['E','I']].iloc[0]

# Parameters
beta = 0.5
sigma = 0.2
gamma = 0.143
N = 10000
mean_degree = 10

# Calculate theoretical R0 assuming ER network (since no network data)
R0 = beta * mean_degree / gamma

max_time, final_data.to_dict(), peak_time, peak_I, epidemic_start_time, epidemic_end_time, epidemic_duration, initial_seeds.to_dict(), R0