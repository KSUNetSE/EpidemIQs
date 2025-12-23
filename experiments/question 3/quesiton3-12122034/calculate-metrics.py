
import numpy as np

# Extract peak infected info
max_I_index = data['SIR_I'].idxmax()
max_I = data.loc[max_I_index, 'SIR_I']
max_I_lower = data.loc[max_I_index, 'SIR_I_90ci_lower']
max_I_upper = data.loc[max_I_index, 'SIR_I_90ci_upper']
max_I_sd_approx = (max_I_upper - max_I_lower) / (2 * 1.645)  # Approximate SD from 90% CI

# Extract time to peak infected
time_to_peak = data.loc[max_I_index, 'time']

# Extract final epidemic size from recovered individuals at last time point
final_R = data.loc[data.index[-1], 'SIR_R']
final_R_lower = data.loc[data.index[-1], 'SIR_R_90ci_lower']
final_R_upper = data.loc[data.index[-1], 'SIR_R_90ci_upper']
final_R_sd_approx = (final_R_upper - final_R_lower) / (2 * 1.645)  

# Calculate duration as the last time when infected > 1 individual
infected_above_one = data[data['SIR_I'] > 1]
duration = infected_above_one['time'].max() if not infected_above_one.empty else 0

# Early fade-out check: final recovered < 10 cases
# Estimate fade-out probability as the proportion of simulations with final R < 10
# Cannot do directly, so approximate with probability from confidence interval
# If the lower bound of final R 90ci is < 10, implies some probability of fade out
fade_out_possible = final_R_lower < 10

# Store results
results = {
    'max_I_mean': max_I,
    'max_I_sd_approx': max_I_sd_approx,
    'time_to_peak_mean': time_to_peak,
    'final_R_mean': final_R,
    'final_R_sd_approx': final_R_sd_approx,
    'duration_mean': duration,
    'fade_out_possible': fade_out_possible
}
results