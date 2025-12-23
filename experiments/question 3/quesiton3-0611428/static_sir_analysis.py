
# Static SIR data
N_12 = results_12.iloc[0][['S', 'I', 'R']].sum()

# Final epidemic size (final R / N)
final_R_12 = results_12['R'].iloc[-1]
final_epidemic_size_12 = final_R_12 / N_12

# Peak prevalence and time of peak
peak_I_12 = results_12['I'].max()
time_peak_I_12 = results_12.loc[results_12['I'].idxmax(), 'time']
peak_prevalence_12 = peak_I_12 / N_12

# Duration (max time with I > 0)
duration_12 = results_12.loc[results_12['I'] > 0, 'time'].max()

# Initial doubling time - fitting exponential growth to early phase
# Use first 5 points for fitting early growth phase
early_time_12 = results_12['time'].values[:5]
early_I_12 = results_12['I'].values[:5]

# Ensure values are positive for log
positive_mask_12 = early_I_12 > 0
early_time_fit_12 = early_time_12[positive_mask_12]
early_I_fit_12 = early_I_12[positive_mask_12]

# Fit exponential curve for initial doubling time
a_init_12 = early_I_fit_12[0]
b_init_12 = 0.5

params_12, _ = curve_fit(exponential_growth, early_time_fit_12, early_I_fit_12, p0=[a_init_12, b_init_12])
tau_12 = np.log(2) / params_12[1]  # doubling time

final_epidemic_size_12, peak_prevalence_12, time_peak_I_12, duration_12, tau_12, N_12