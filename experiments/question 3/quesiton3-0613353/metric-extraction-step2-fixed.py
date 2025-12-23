
# Fix multiplication by N for results-11.csv metrics

final_R_mean_11 = data_11['R_mean'].iloc[-1] * N
final_R_std_11 = data_11['R_std'].iloc[-1] * N

peak_I_fraction_11 = data_11['I_mean'].max()
peak_I_time_11 = data_11.loc[data_11['I_mean'].idxmax(), 't']
peak_I_count_11 = peak_I_fraction_11 * N

metrics_11_fixed = {
    'final_R_mean': final_R_mean_11,  # individuals
    'final_R_std': final_R_std_11,    # individuals
    'peak_I_fraction': peak_I_fraction_11,  # fraction of population
    'peak_I_count': peak_I_count_11,  # individuals
    'peak_I_time': peak_I_time_11,     # time (step)
    'time_to_peak': peak_I_time_11,   # time (step)
    'epidemic_duration': epidemic_duration_11  # time (step)
}
metrics_11_fixed