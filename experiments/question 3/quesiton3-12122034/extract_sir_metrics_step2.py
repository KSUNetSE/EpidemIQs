
# Threshold to count active infection presence
infected_thresh = 1

duration_indices = np.where(I > infected_thresh)[0]
if len(duration_indices) == 0:
    total_duration = 0
else:
    total_duration = times[duration_indices[-1]]  # time units

# For doubling time, take early times where infection > threshold
# Pick first 5 points for fit, if available
early_indices = duration_indices[:5]
early_times = times[early_indices]
early_infected = I[early_indices]

if len(early_infected) >= 2 and np.all(early_infected > 0):
    log_infected = np.log(early_infected)
    coeffs = np.polyfit(early_times, log_infected, 1)
    growth_rate = coeffs[0]
    doubling_time = np.log(2) / growth_rate if growth_rate > 0 else np.nan
else:
    doubling_time = np.nan

total_duration, doubling_time