
# The static dataset has many more rows indicating repeated measurements or aggregated data points across realizations
# Extract final epidemic size metrics from last time point
final_time_static = results_static['time'].max()

# Filter rows corresponding to the final time
final_rows_static = results_static[results_static['time'] == final_time_static]

# Calculate mean final recovered fraction and 90% CI bounds
# Since R is count, convert to fraction of N=1000
N = 1000
final_size_mean_static = final_rows_static['R'].mean() / N
final_size_90ci_lower_static = final_rows_static['R_90ci_lower'].mean() / N
final_size_90ci_upper_static = final_rows_static['R_90ci_upper'].mean() / N

# To find peak prevalence and time to peak, find max mean I and corresponding time
max_I_index = results_static['I'].idxmax()
peak_prev_mean_static = results_static.loc[max_I_index, 'I'] / N
# Time of peak
t_peak_mean_static = results_static.loc[max_I_index, 'time']

# Peak I 90% CI at time of peak
peak_prev_90ci_lower_static = results_static.loc[max_I_index, 'I_90ci_lower'] / N
peak_prev_90ci_upper_static = results_static.loc[max_I_index, 'I_90ci_upper'] / N

final_size_mean_static, final_size_90ci_lower_static, final_size_90ci_upper_static, peak_prev_mean_static, peak_prev_90ci_lower_static, peak_prev_90ci_upper_static, t_peak_mean_static