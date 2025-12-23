
# We found B(t) and its 90 ci lower and upper bounds in results-11.csv
# Now let's extract peak prevalence in B, peak time, and duration of epidemic for B
results_11['B_peak'] = results_11['B'].max()
results_11_peak_time = results_11.loc[results_11['B'].idxmax(), 'time']

# Duration of epidemic in B: time from first non-zero B to last non-zero B
b_nonzero_times = results_11.loc[results_11['B'] > 0, 'time']
epidemic_duration = b_nonzero_times.max() - b_nonzero_times.min() if len(b_nonzero_times) > 1 else 0

# Width of 90% CI for B(t) at peak
peak_idx = results_11['B'].idxmax()
b_90ci_width_at_peak = results_11.loc[peak_idx, 'B_90ci_upper'] - results_11.loc[peak_idx, 'B_90ci_lower']

peak_b_metrics = {
    'peak_B': results_11['B_peak'],
    'peak_time_B': results_11_peak_time,
    'duration_B': epidemic_duration,
    'B_90ci_width_at_peak': b_90ci_width_at_peak
}
peak_b_metrics