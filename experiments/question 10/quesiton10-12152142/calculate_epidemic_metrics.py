
import numpy as np

def calculate_doubling_time(times, values):
    # Fit exponential growth to early data: log(values) = a*time + b
    # Use the first 10 data points where I>1
    mask = values > 1
    times_growth = times[mask]
    values_growth = values[mask]
    if len(times_growth) < 2:
        return np.nan
    times_growth = times_growth[:10]
    values_growth = values_growth[:10]
    
    log_vals = np.log(values_growth)
    # Linear regression
    A = np.vstack([times_growth, np.ones(len(times_growth))]).T
    slope, intercept = np.linalg.lstsq(A, log_vals, rcond=None)[0]
    doubling_time = np.log(2) / slope if slope > 0 else np.nan
    return doubling_time

N = 1000

# Extract columns
times = data_detailed['time'].values
S = data_detailed['S'].values
I = data_detailed['I'].values
R = data_detailed['R'].values

# Peak prevalence
peak_prevalence_num = np.max(I)
peak_prevalence_pct = peak_prevalence_num / N * 100
peak_index = np.argmax(I)

# Time to peak
time_to_peak = times[peak_index]

# Final epidemic size (total R at last time point)
final_epidemic_size_num = R[-1]
final_epidemic_size_pct = final_epidemic_size_num / N * 100

# Epidemic duration: from first I >1 to last I <1
first_infected_index = np.argmax(I > 1)  # first index where I >1
# last index where after peak I < 1
post_peak_indices = np.where(times > time_to_peak)[0]
last_infected_index = -1
for idx in post_peak_indices:
    if I[idx] < 1:
        last_infected_index = idx
        break
# If last_infected_index not found, use last time point
if last_infected_index == -1:
    last_infected_index = len(times) - 1

epidemic_duration = times[last_infected_index] - times[first_infected_index]

# Initial doubling time
initial_doubling_time = calculate_doubling_time(times, I)

metrics = {
    'Peak Prevalence (number)': peak_prevalence_num,
    'Peak Prevalence (%)': peak_prevalence_pct,
    'Time to Peak (days)': time_to_peak,
    'Final Epidemic Size (number)': final_epidemic_size_num,
    'Final Epidemic Size (%)': final_epidemic_size_pct,
    'Epidemic Duration (days)': epidemic_duration,
    'Initial Doubling Time (days)': initial_doubling_time,
}
metrics