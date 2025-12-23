
import numpy as np

# Extract peak D and time at peak for core dataset
peak_D_idx_core = df_core['D'].idxmax()
peak_D_core = df_core.loc[peak_D_idx_core, 'D']
peak_time_core = df_core.loc[peak_D_idx_core, 'time']
peak_D_lower_core = df_core.loc[peak_D_idx_core, 'D_90ci_lower']
peak_D_upper_core = df_core.loc[peak_D_idx_core, 'D_90ci_upper']

# Final L at simulation end (last time point)
final_L_core = df_core.loc[df_core.index[-1], 'L']
final_L_lower_core = df_core.loc[df_core.index[-1], 'L_90ci_lower']
final_L_upper_core = df_core.loc[df_core.index[-1], 'L_90ci_upper']

# Crisis duration: time from start until D returns to minimum (close to zero) after peak
# Find the time after peak where D is minimal (closest to zero) or zero
post_peak_D_core = df_core.loc[peak_D_idx_core:, 'D']
min_D_idx_core = post_peak_D_core.idxmin()
crisis_duration_core = df_core.loc[min_D_idx_core, 'time'] - df_core.loc[0, 'time']

# Area under the curve for D using trapezoidal rule
area_D_core = np.trapz(df_core['D'], df_core['time'])

(peak_D_core, peak_time_core, final_L_core, crisis_duration_core, area_D_core, peak_D_lower_core, peak_D_upper_core, final_L_lower_core, final_L_upper_core)