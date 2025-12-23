
import numpy as np
from scipy.optimize import curve_fit

# Constants
def exponential_growth(t, a, b):
    return a * np.exp(b * t)

# Temporal SIR data
N_11 = results_11.iloc[0][['S_mean', 'I_mean', 'R_mean']].sum()

# Final epidemic size (mean and std)
final_R_mean_11 = results_11['R_mean'].iloc[-1]
final_R_std_11 = results_11['R_std'].iloc[-1]
final_epidemic_size_mean_11 = final_R_mean_11 / N_11
final_epidemic_size_std_11 = final_R_std_11 / N_11

# Peak prevalence and time of peak
peak_I_mean_11 = results_11['I_mean'].max()
time_peak_I_11 = results_11.loc[results_11['I_mean'].idxmax(), 'time']
peak_prevalence_11 = peak_I_mean_11 / N_11

# Duration (max time with I_mean>0)
duration_11 = results_11.loc[results_11['I_mean'] > 0, 'time'].max()

# Initial doubling time - fitting exponential growth to early phase
# Use first 5 points for fitting early growth
early_time = results_11['time'].values[:5]
early_I_mean = results_11['I_mean'].values[:5]

# Ensure values are positive for log
positive_mask = early_I_mean > 0
early_time_fit = early_time[positive_mask]
early_I_fit = early_I_mean[positive_mask]

# Fit exponential curve
a_init = early_I_fit[0]
b_init = 0.5

params, _ = curve_fit(exponential_growth, early_time_fit, early_I_fit, p0=[a_init, b_init])
tau = np.log(2) / params[1]  # doubling time

# Print the calculated variables
final_epidemic_size_mean_11, final_epidemic_size_std_11, peak_prevalence_11, time_peak_I_11, duration_11, tau, N_11