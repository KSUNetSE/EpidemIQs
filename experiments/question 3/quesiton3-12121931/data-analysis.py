
# Data covers time series of S, I, R with 90% confidence intervals
# Size is 2769 rows, indicating many time points
# Next, extract the requested metrics:

# (1) Final epidemic size as total and fraction of R at final time
final_time = data['time'].iloc[-1]
final_R = data['R'].iloc[-1]
total_population = data['S'].iloc[0] + data['I'].iloc[0] + data['R'].iloc[0]
fraction_final_R = final_R / total_population

# (2) Peak infection rate (max I) and corresponding time
peak_I = data['I'].max()
time_peak_I = data.loc[data['I'] == peak_I, 'time'].iloc[0]

# (3) Epidemic duration (time until I falls below 1 or ends)
# Find the last time when I >= 1
infected_above_one = data[data['I'] >= 1]
epidemic_end_time = infected_above_one['time'].iloc[-1] if not infected_above_one.empty else 0

def init_outbreak(d):
    # Determining outbreak based on last value of R
    return d['R'].iloc[-1] > 10

outbreak_occurred = init_outbreak(data)

# (4) Outbreak probability requires multiple runs data; 
# Since this file is one run, we interpret presence or absence of outbreak for this single run
outbreak_probability = 1 if outbreak_occurred else 0

# (5) Brief 1-line summary of the overall shape of the epidemic curve
if peak_I <= 1:
    epidemic_curve_summary = "Rapid die-out with no significant outbreak"
elif final_R > 0.5 * total_population:
    epidemic_curve_summary = "Large outbreak with a single broad peak"
else:
    epidemic_curve_summary = "Moderate outbreak with a peak and steady decline"

# (6) Additional observations
# Calculate time to reach peak
time_to_peak = time_peak_I - data['time'].iloc[0]

# Check if R stabilizes at the end
R_stabilization = abs(data['R'].iloc[-1] - data['R'].iloc[-10]) < 1e-3

metrics = {
    'final_time': final_time,
    'final_R': final_R,
    'fraction_final_R': fraction_final_R,
    'peak_I': peak_I,
    'time_peak_I': time_peak_I,
    'epidemic_end_time': epidemic_end_time,
    'outbreak_occurred': outbreak_occurred,
    'outbreak_probability': outbreak_probability,
    'epidemic_curve_summary': epidemic_curve_summary,
    'time_to_peak': time_to_peak,
    'R_stabilization': R_stabilization
}