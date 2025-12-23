
# Repeat extraction for results-11.csv

final_R_mean_11 = data_11['R_mean'].iloc[-1] * N
final_R_std_11 = data_11['R_std'].iloc[-1] * N

peak_I_fraction_11 = data_11['I_mean'].max()
peak_I_time_11 = data_11.loc[data_11['I_mean'].idxmax(), 't']
peak_I_count_11 = peak_I_fraction_11 * N

time_to_peak_11 = peak_I_time_11

I_zero_times_11 = data_11[data_11['I_mean'] <= 0].t
if not I_zero_times_11.empty:
    epidemic_duration_11 = I_zero_times_11.iloc[0]
else:
    epidemic_duration_11 = data_11['t'].iloc[-1]  # no zero found

metrics_11 = {
    'final_R_mean': final_R_mean_11,  # individuals
    'final_R_std': final_R_std_11,    # individuals
    'peak_I_fraction': peak_I_fraction_11,  # fraction of population
    'peak_I_count': peak_I_count_11,  # individuals
    'peak_I_time': peak_I_time_11,     # time (step)
    'time_to_peak': time_to_peak_11,   # time (step)
    'epidemic_duration': epidemic_duration_11  # time (step)
}
metrics_11