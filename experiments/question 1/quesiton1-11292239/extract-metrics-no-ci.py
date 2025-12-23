
# Extract values
N = 2000
time = data['time'].values
I = data['I'].values / N
R = data['R'].values / N

# 1. Epidemic Duration: last time with I >= 1/N
threshold_normalized = 1 / N
indices = np.where(I >= threshold_normalized)[0]
epidemic_duration = time[indices[-1]] if len(indices) > 0 else np.nan

# 2. Peak I and time
peak_idx = np.argmax(I)
peak_I = I[peak_idx]
peak_time = time[peak_idx]

# 3. Final R at last time
final_R = R[-1]

# 4. Initial doubling time: find the first time I doubles from initial
initial_I = I[0]
doubling_candidates = np.where(I >= 2 * initial_I)[0]
doubling_time = (time[doubling_candidates[0]] - time[0]) if len(doubling_candidates) > 0 else np.nan

metrics = {
    'epidemic_duration': epidemic_duration,  # same time units as in data
    'peak_I': peak_I,  # normalized by N
    'peak_time': peak_time,  # same time units
    'final_R': final_R,  # normalized by N
    'initial_doubling_time': doubling_time  # same time units
}
metrics