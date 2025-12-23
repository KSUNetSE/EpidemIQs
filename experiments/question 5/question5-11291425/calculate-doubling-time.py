
import numpy as np
from scipy.stats import linregress

# Use early phase data, choosing time window from 0 to 0.8 manually based on general understanding
early_phase = data[(data['time'] >= 0) & (data['time'] <= 0.8)].copy()

# To avoid log(0), filter out any zero or negative values
early_phase = early_phase[early_phase['I'] > 0]

# Calculate log(I)
early_phase['log_I'] = np.log(early_phase['I'])

# Perform linear regression on log(I) ~ time
slope, intercept, r_value, p_value, std_err = linregress(early_phase['time'], early_phase['log_I'])

# Exponential growth rate r = slope
r = slope

# Calculate doubling time Td = ln(2) / r if r > 0
if r > 0:
    doubling_time = np.log(2) / r
else:
    doubling_time = np.nan

r, doubling_time