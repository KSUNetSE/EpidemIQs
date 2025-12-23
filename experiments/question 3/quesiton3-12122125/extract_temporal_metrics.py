
# Extract summary metrics and CIs from temporal data
final_size_mean_temporal = results_temporal.loc[0, 'final_size_mean']
final_size_5p_temporal = results_temporal.loc[0, 'final_size_5p']
final_size_95p_temporal = results_temporal.loc[0, 'final_size_95p']

peak_prev_mean_temporal = results_temporal.loc[0, 'peak_prev_mean']
peak_prev_5p_temporal = results_temporal.loc[0, 'peak_prev_5p']
peak_prev_95p_temporal = results_temporal.loc[0, 'peak_prev_95p']

# Time to peak information
# Note: there are some NaNs so we find first non-NaN row to extract these 
idx = results_temporal['t_peak_mean'].first_valid_index()
t_peak_mean_temporal = results_temporal.loc[idx, 't_peak_mean']
t_peak_5p_temporal = results_temporal.loc[idx, 't_peak_5p']
t_peak_95p_temporal = results_temporal.loc[idx, 't_peak_95p']

# Extract mean time series and 90% CI for S, I, R (5p and 95p percentiles)
time_series_temporal = results_temporal[['time', 'S_mean', 'S_5p', 'S_95p', 'I_mean', 'I_5p', 'I_95p', 'R_mean', 'R_5p', 'R_95p']]

final_size_mean_temporal, final_size_5p_temporal, final_size_95p_temporal, peak_prev_mean_temporal, peak_prev_5p_temporal, peak_prev_95p_temporal, t_peak_mean_temporal, t_peak_5p_temporal, t_peak_95p_temporal