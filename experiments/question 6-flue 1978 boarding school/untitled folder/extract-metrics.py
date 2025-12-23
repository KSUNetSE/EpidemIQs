
# Extract columns needed
B = data['B']
C = data['C']
R = data['R']
time = data['time']

# (1) Peak value and time for B(t)
peak_B_value = B.max()
peak_B_time = time[B.idxmax()]

# (2) Final cumulative Attack Rate AR = fraction ever in B or R at end of simulation
final_B = B.iloc[-1]
final_R = R.iloc[-1]
N = 763  # total population from initial condition
# cumulative fraction ever in B or R at end
final_AR = (final_B + final_R) / N

# (3) Main epidemic duration (first to last day B or C > 1 individual)
# find first and last time where B or C > 1
threshold = 1
B_C_gt_threshold = (B > threshold) | (C > threshold)
if B_C_gt_threshold.any():
    first_day = time[B_C_gt_threshold.idxmax()]  # first true index gives first day
    last_day = time[len(B_C_gt_threshold) - 1 - B_C_gt_threshold[::-1].idxmax()]  # last true index
    epidemic_duration = last_day - first_day
else:
    first_day = None
    last_day = None
    epidemic_duration = 0

# (4) Full time series for plotting
B_time_series = B.values
C_time_series = C.values

peak_B_value, peak_B_time, final_AR, epidemic_duration, first_day, last_day, len(B_time_series), len(C_time_series)