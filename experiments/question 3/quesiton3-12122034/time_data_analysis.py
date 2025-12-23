
# Check unique time points and their counts
unique_times = data['time'].unique()
unique_time_count = len(unique_times)

# Check if time restarts at any point (for segmentation by realization)
time_diff = data['time'].diff()
time_reset_indices = time_diff[time_diff < 0].index.tolist()  # Points where time decreases, possible split points

# If the number of unique times matches data length, likely single time series, else multiple runs concatenated

unique_time_count, len(time_diff), len(time_reset_indices)