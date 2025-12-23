
import numpy as np

# Extract time and values
time = data['time'].values
susceptible = data['SIR_S'].values
infected = data['SIR_I'].values
recovered = data['SIR_R'].values

# We hypothesize final epidemic size approximately R(t=last) since recovered+infected+susceptible=N
final_size = recovered[-1]  # last time point recovered

# Peak prevalence (max I)
peak_I = np.max(infected)
peak_time = time[np.argmax(infected)]

# Epidemic duration: time until the infected fall below 1 or very low threshold (e.g. 1)
threshold = 1
infected_below_threshold = np.where(infected < threshold)[0]
if len(infected_below_threshold) > 0:
    epi_duration = time[infected_below_threshold[0]]
else:
    epi_duration = time[-1]  # If never fall below threshold in data

# For early fade-out, we would need per-run data; here we can estimate fraction with final size < 10
# Without individual runs data, this can't be computed directly. We'll return None for now.
early_fadeout_prob = None

# Doubling time: estimate early slope
# Take early phase (say first 10 time points), filter to infected > 0
early_time = time[:10]
early_infected = infected[:10]

# Calculate growth rate via linear regression on log of infected
from scipy.stats import linregress
mask = early_infected > 0
if np.sum(mask) > 1:
    slope, intercept, r_value, p_value, std_err = linregress(early_time[mask], np.log(early_infected[mask]))
    doubling_time = np.log(2) / slope if slope > 0 else None
else:
    doubling_time = None

metrics = {
    'final_epidemic_size_mean': final_size,
    # SD, median, quantiles unavailable without per-run data
    'peak_prevalence_mean': peak_I,
    'time_to_peak_mean': peak_time,
    'epidemic_duration_mean': epi_duration,
    'early_fadeout_probability': early_fadeout_prob,
    'doubling_time': doubling_time
}
metrics