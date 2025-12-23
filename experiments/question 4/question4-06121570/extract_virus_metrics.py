
import numpy as np

# Constants
N = 1000
threshold = 0.01 * N  # 10

# Getting the last 20 time points
last_20 = data.tail(20)

# Helper function to calculate metrics for a given virus column

def calculate_metrics(virus_col):
    virus_data = data[virus_col]
    times = data['time']

    # Final prevalence: average of last 20 time steps
    final_prevalence = last_20[virus_col].mean()

    # Peak prevalence and time
    peak_idx = virus_data.idxmax()
    peak_prevalence = virus_data[peak_idx]
    peak_time = times[peak_idx]

    # Time to extinction: first time prevalence drops below threshold
    extinct_times = times[virus_data < threshold]
    time_to_extinction = extinct_times.iloc[0] if not extinct_times.empty else None

    # Dominance indicator: 1 if final prevalence > threshold, else 0
    dominance = 1 if final_prevalence > threshold else 0

    return {
        'final_prevalence': final_prevalence,
        'peak_prevalence': peak_prevalence,
        'peak_time': peak_time,
        'time_to_extinction': time_to_extinction,
        'dominance': dominance
    }

metrics_I1 = calculate_metrics('I1')
metrics_I2 = calculate_metrics('I2')

# Calculate coexistence duration where both prevalence are > threshold
coexistence_mask = (data['I1'] > threshold) & (data['I2'] > threshold)
coexistence_times = data.loc[coexistence_mask, 'time']

# Coexistence duration in time units (consider contiguous periods)
coexistence_duration = 0
if not coexistence_times.empty:
    # Find gaps between consecutive time points beyond some tolerance (0.1 units here)
    gaps = coexistence_times.diff().fillna(0) > 0.1
    # Split into contiguous segments
    segments = coexistence_times[~gaps].groupby(gaps.cumsum())
    # Sum durations of each contiguous segment
    coexistence_duration = sum(seg.max() - seg.min() for _, seg in segments)

metrics_I1, metrics_I2, coexistence_duration
import numpy as np

N = 1000
threshold = 0.01 * N

# Calculate final prevalence (average of last 20 time steps)
final_I1 = np.mean(data['I1'].tail(20))
final_I2 = np.mean(data['I2'].tail(20))

# Find peak prevalence and time of peak
peak_I1 = data['I1'].max()
peak_time_I1 = data['time'][data['I1'].idxmax()]

peak_I2 = data['I2'].max()
peak_time_I2 = data['time'][data['I2'].idxmax()]

# Calculate time to extinction (prevalence < 0.01 * N)
time_to_extinct_I1 = None
below_threshold_I1 = data[data['I1'] < threshold]
if not below_threshold_I1.empty:
    time_to_extinct_I1 = below_threshold_I1.iloc[0]['time']


time_to_extinct_I2 = None
below_threshold_I2 = data[data['I2'] < threshold]
if not below_threshold_I2.empty:
    time_to_extinct_I2 = below_threshold_I2.iloc[0]['time']

# Dominance/Coexistence indicator (1 if final prevalence > 0.01 * N)
dominance_I1 = 1 if final_I1 > threshold else 0
dominance_I2 = 1 if final_I2 > threshold else 0

# Check coexistence duration - time intervals where both I1 and I2 > threshold
dual_prevalence = (data['I1'] > threshold) & (data['I2'] > threshold)

# Find contiguous True segments and sum their durations
def contiguous_duration(bool_series, times):
    durations = []
    start_idx = None
    for i, val in enumerate(bool_series):
        if val and start_idx is None:
            start_idx = i
        elif not val and start_idx is not None:
            durations.append(times[i-1] - times[start_idx])
            start_idx = None
    if start_idx is not None:
        durations.append(times.iloc[-1] - times[start_idx])
    return sum(durations)

coexistence_duration = contiguous_duration(dual_prevalence, data['time'])

metrics = {
    "I1": {
        "final_prevalence": final_I1,  # individuals
        "peak_prevalence": peak_I1,  # individuals
        "peak_time": peak_time_I1,  # time units
        "time_to_extinction": time_to_extinct_I1,  # time units or None
        "dominance_indicator": dominance_I1,  # binary
    },
    "I2": {
        "final_prevalence": final_I2,
        "peak_prevalence": peak_I2,
        "peak_time": peak_time_I2,
        "time_to_extinction": time_to_extinct_I2,
        "dominance_indicator": dominance_I2,
    },
    "coexistence_duration": coexistence_duration  # time units
}
metrics