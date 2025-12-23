
import numpy as np

N = 10000  # assumed total population to convert fractions to counts

# Final epidemic size (R_mean and R_std at final time)
final_R_mean = data['R_mean'].iloc[-1] * N
final_R_std = data['R_std'].iloc[-1] * N

# Peak infection rate
peak_I_fraction = data['I_mean'].max()
peak_I_time = data.loc[data['I_mean'].idxmax(), 't']
peak_I_count = peak_I_fraction * N

# Time to peak is same as peak_I_time

time_to_peak = peak_I_time

# Epidemic duration: time when I_mean first hits 0 after initialization
# Find first t where I_mean is 0 after start time
I_zero_times = data[data['I_mean'] <= 0].t
if not I_zero_times.empty:
    epidemic_duration = I_zero_times.iloc[0]
else:
    epidemic_duration = data['t'].iloc[-1]  # no zero found

metrics = {
    'final_R_mean': final_R_mean,  # individuals
    'final_R_std': final_R_std,    # individuals
    'peak_I_fraction': peak_I_fraction,  # fraction of population
    'peak_I_count': peak_I_count,  # individuals
    'peak_I_time': peak_I_time,     # time (step)
    'time_to_peak': time_to_peak,   # time (step)
    'epidemic_duration': epidemic_duration  # time (step)
}
metrics