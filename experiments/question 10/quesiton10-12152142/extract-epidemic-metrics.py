
import numpy as np

N = 1000  # total population

# Peak prevalence (max I) and percent
peak_I = detailed_data['I'].max()
peak_I_time = detailed_data.loc[detailed_data['I'].idxmax(), 'time']
peak_I_percent = peak_I / N * 100

peak_I_CI_lower = detailed_data['I_90ci_lower'].max()
peak_I_CI_upper = detailed_data['I_90ci_upper'].max()

# Final epidemic size (final R) and percent
final_R = detailed_data['R'].iloc[-1]
final_R_percent = final_R / N * 100
final_R_CI_lower = detailed_data['R_90ci_lower'].iloc[-1]
final_R_CI_upper = detailed_data['R_90ci_upper'].iloc[-1]

# Epidemic duration: time between first I > 1 and last I > 1
I_above_1 = detailed_data[detailed_data['I'] > 1]
if not I_above_1.empty:
    epidemic_start = I_above_1['time'].iloc[0]
    epidemic_end = I_above_1['time'].iloc[-1]
    epidemic_duration = epidemic_end - epidemic_start
else:
    epidemic_start = np.nan
    epidemic_end = np.nan
    epidemic_duration = 0

# Additional speed/severity metrics: time to R reaching 10% and 20%
time_to_10pct_R = detailed_data[detailed_data['R'] >= 0.1 * N]['time'].iloc[0] if not detailed_data[detailed_data['R'] >= 0.1 * N].empty else np.nan
time_to_20pct_R = detailed_data[detailed_data['R'] >= 0.2 * N]['time'].iloc[0] if not detailed_data[detailed_data['R'] >= 0.2 * N].empty else np.nan

metrics = {
    'Peak Prevalence (number)': peak_I,
    'Peak Prevalence (%)': peak_I_percent,
    'Peak Prevalence 90% CI Lower': peak_I_CI_lower,
    'Peak Prevalence 90% CI Upper': peak_I_CI_upper,
    'Time to Peak (days)': peak_I_time,
    'Final Epidemic Size (number)': final_R,
    'Final Epidemic Size (%)': final_R_percent,
    'Final Epidemic Size 90% CI Lower': final_R_CI_lower,
    'Final Epidemic Size 90% CI Upper': final_R_CI_upper,
    'Epidemic Duration (days)': epidemic_duration,
    'Epidemic Start Time (days)': epidemic_start,
    'Epidemic End Time (days)': epidemic_end,
    'Time to 10% R (days)': time_to_10pct_R,
    'Time to 20% R (days)': time_to_20pct_R
}

metrics