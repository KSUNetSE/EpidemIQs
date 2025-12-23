
from scipy.stats import linregress

# Filter early phase: where mean I is > 0 and before peak time
early_phase = data[(data['time'] < t_peak) & (data['I'] > 0)]

# Linear fit to log(I) to estimate exponential growth rate
log_I = np.log(early_phase['I'])
slope, intercept, r_value, p_value, std_err = linregress(early_phase['time'], log_I)

growth_rate = slope
# Doubling time = ln(2) / growth rate
# Check if growth_rate > 0 to avoid divide by zero or negative
if growth_rate > 0:
    doubling_time = np.log(2) / growth_rate
else:
    doubling_time = np.nan

metrics['doubling_time'] = doubling_time
metrics['doubling_fit_r2'] = r_value**2
metrics