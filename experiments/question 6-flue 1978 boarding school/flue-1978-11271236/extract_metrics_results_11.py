
# Constants
N = 763

# Extract metrics from results-11 data
data = data_11

final_time = data['time'].iloc[-1]
final_B = data['B'].iloc[-1]
final_C = data['C'].iloc[-1]
final_attack_rate = (final_B + final_C) / N * 100

peak_B = data['B'].max()
peak_B_time = data.loc[data['B'].idxmax(), 'time']

# Epidemic duration (last day B>0)
nonzero_B_times = data.loc[data['B'] > 0, 'time']
epidemic_duration = nonzero_B_times.max() if not nonzero_B_times.empty else 0

# Mean duration in B (time duration B > half max)
half_max_threshold = peak_B / 2
above_half_max = data.loc[data['B'] > half_max_threshold, 'time']
mean_duration_B = above_half_max.max() - above_half_max.min() if not above_half_max.empty else 0

# Final C count
final_C_count = final_C

# Metadata for parameters - to be checked if available (Not in main columns, check csv head for metadata)
# We don't see columns for β, σ, κ, γ, R0 in these data frames, so need another way to extract these if metadata exists.

# Prepare results for this file
metrics_11 = {
    'Final attack rate (%)': final_attack_rate,
    'Peak Bed Occupancy (max B)': peak_B,
    'Peak Bed Timing (day)': peak_B_time,
    'Epidemic Duration (days)': epidemic_duration,
    'Mean Duration in B (days)': mean_duration_B,
    'Final C count': final_C_count
}
metrics_11