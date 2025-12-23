
import pandas as pd

# Load data from the specified file path
file_path = 'output/results-21.csv'
data = pd.read_csv(file_path)

# Display basic information about the data to understand its structure and columns
info = data.info()

# Display the first few rows to get a glimpse of the data
head = data.head()

info, head
# Check the columns again to confirm what they are (case sensitive and whitespaces)
col_list = data.columns.tolist()
col_list
# Check the initial values of the population compartments
initial_values = data.iloc[0][['S_mean', 'I_mean', 'R_mean']]

# Assuming total population N is sum of S, I, R at initial time
N = initial_values.sum()

initial_values, N
import numpy as np

# Extract final epidemic size (mean and SD) from last row
final_R_mean = data['R_mean'].iloc[-1]
final_R_std = data['R_std'].iloc[-1]

# Find peak infection rate and the time it occurs
peak_I_mean = data['I_mean'].max()
time_of_peak = data.loc[data['I_mean'].idxmax(), 't']

# Time to peak
# Time step of peak I_mean (continuous time).
time_to_peak = time_of_peak

# Epidemic duration: time when I_mean first hits zero after start
# Find first time I_mean is zero after t=0 (if any)
post_peak_data = data[data['t'] > time_to_peak]
I_zero_times = post_peak_data[post_peak_data['I_mean'] <= 1e-6]  # threshold close to zero

epidemic_duration = I_zero_times['t'].min() if not I_zero_times.empty else data['t'].max()

# Prepare results
metrics = {
    'final_R_mean': final_R_mean,
    'final_R_std': final_R_std,
    'peak_I_mean': peak_I_mean,
    'time_of_peak': time_of_peak,
    'time_to_peak': time_to_peak,
    'epidemic_duration': epidemic_duration
}
metrics