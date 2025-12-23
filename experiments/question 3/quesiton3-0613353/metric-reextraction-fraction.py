
# Re-extract metrics for results-01.csv as fractions for clear reporting
metrics_fraction = {
    'final_R_mean_fraction': data['R_mean'].iloc[-1],  # fraction of pop
    'final_R_std_fraction': data['R_std'].iloc[-1],
    'peak_I_fraction': data['I_mean'].max(),
    'peak_I_time': data.loc[data['I_mean'].idxmax(), 't'],
    'time_to_peak': data.loc[data['I_mean'].idxmax(), 't'],
    'epidemic_duration': epidemic_duration
}