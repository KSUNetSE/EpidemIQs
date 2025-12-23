
# Based on the data structure observed from head(), the columns are time, S, I, R (with confidence intervals)
# N = 10,000 total population
# We want to extract metrics:
# 1. Final size (max R/N)
# 2. Peak infection prevalence (max I/N)
# 3. Time of peak (time when max I/N occurs)
# 4. Epidemic duration (time when I falls back to near 0, e.g., below 1 individual)
# 5. Doubling time (early phase if measurable) - calculate doubling time from early exponential phase of I

# Calculate fractions
N = 10000

# Normalize columns by N for fraction values
data['S_frac'] = data['S'] / N
data['I_frac'] = data['I'] / N
data['R_frac'] = data['R'] / N

# 1. Final size = max(R/N)
final_size = data['R_frac'].iloc[-1]  # last value of R_normalized

# 2. Peak infection prevalence = max(I/N)
peak_infection_prevalence = data['I_frac'].max()

# 3. Time of peak infection prevalence
time_of_peak = data['time'][data['I_frac'].idxmax()]

# 4. Epidemic duration: time until I falls below a threshold, e.g. 1 individual infected (I < 1/N=0.0001)
# Since time is sorted, find last time where I_fraction > 1/N then duration is that time
threshold = 1 / N
infected_above_threshold = data[data['I_frac'] > threshold]
epidemic_duration = infected_above_threshold['time'].max() - infected_above_threshold['time'].min()

# 5. Doubling time: from early phase, select times when I is small but growing exponentially
# We take the first few points where I_frac is above some small threshold and increasing
small_threshold = 1e-4

# Filter for early phase when I_frac > small_threshold and before peak time
early_phase = data[(data['I_frac'] > small_threshold) & (data['time'] < time_of_peak)]

# To estimate doubling time, perform linear regression on log(I_frac) vs time in early phase
from scipy.stats import linregress
import numpy as np

# Select data for regression
if len(early_phase) > 1 and (early_phase['I_frac'] > 0).all():
    slope, intercept, r_value, p_value, std_err = linregress(early_phase['time'], np.log(early_phase['I_frac']))
    if slope > 0:
        doubling_time = np.log(2) / slope
    else:
        doubling_time = None
else:
    doubling_time = None

final_size, peak_infection_prevalence, time_of_peak, epidemic_duration, doubling_time