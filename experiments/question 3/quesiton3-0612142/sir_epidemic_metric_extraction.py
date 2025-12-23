
# Fix NameError by properly defining 'near_zero_threshold' and rerun calculation of metrics

near_zero_threshold = 0.001

# 1) Final epidemic size = fraction ever infected = max of R fraction
final_epidemic_size = data['R_frac'].max()  # fraction

# 2) Peak infection rate = max infectious fraction and time
peak_infectious_fraction = data['I_frac'].max()
peak_infectious_time = data.loc[data['I_frac'].idxmax(), 'time']

# 3) Epidemic duration = time from first nonzero infected to return to near-zero infected (e.g. I_frac < 0.001)
first_nonzero_I_time = data.loc[data['I_frac'] > 0, 'time'].min()

# time when infectious returns below threshold after peak
post_peak_data = data.loc[data['time'] > peak_infectious_time]
epidemic_end_time_data = post_peak_data.loc[post_peak_data['I_frac'] < near_zero_threshold]
if not epidemic_end_time_data.empty:
    epidemic_end_time = epidemic_end_time_data['time'].min()
else:
    # If never goes below threshold, use last time
    epidemic_end_time = data['time'].max()

epidemic_duration = epidemic_end_time - first_nonzero_I_time

# 4) Outbreak velocity: time to reach half of final epidemic size (R_frac = 0.5*final)
half_final_R = 0.5 * final_epidemic_size
# find first time R_frac reaches or exceeds half final size
reach_half_R = data.loc[data['R_frac'] >= half_final_R, 'time'].min()

final_epidemic_size, peak_infectious_fraction, peak_infectious_time, epidemic_duration, reach_half_R
