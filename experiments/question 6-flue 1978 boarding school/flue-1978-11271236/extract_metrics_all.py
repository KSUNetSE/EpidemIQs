
# Reuse metrics extraction logic for other results data files
files_data = {
    'results-12': data_12,
    'results-13': data_13,
    'results-14': data_14,
    'results-15': data_15,
    'results-16': data_16
}

metrics_all = {}

for key, df in files_data.items():
    final_B = df['B'].iloc[-1]
    final_C = df['C'].iloc[-1]
    final_attack_rate = (final_B + final_C) / N * 100
    peak_B = df['B'].max()
    peak_B_time = df.loc[df['B'].idxmax(), 'time']
    nonzero_B_times = df.loc[df['B'] > 0, 'time']
    epidemic_duration = nonzero_B_times.max() if not nonzero_B_times.empty else 0
    half_max_threshold = peak_B / 2
    above_half_max = df.loc[df['B'] > half_max_threshold, 'time']
    mean_duration_B = above_half_max.max() - above_half_max.min() if not above_half_max.empty else 0
    final_C_count = final_C

    metrics_all[key] = {
        'Final attack rate (%)': final_attack_rate,
        'Peak Bed Occupancy (max B)': peak_B,
        'Peak Bed Timing (day)': peak_B_time,
        'Epidemic Duration (days)': epidemic_duration,
        'Mean Duration in B (days)': mean_duration_B,
        'Final C count': final_C_count
    }

metrics_all