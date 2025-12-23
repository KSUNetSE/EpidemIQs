
from scipy.optimize import curve_fit

# Function for exponential growth
exp_growth = lambda t, a, b: a * np.exp(b * t)

# Restrict initial phase: we expect initial exponential growth up to peak or some initial portion, let's take I >1 and first 15 days max
init_phase = data[(data['time'] <= 15) & (data['I'] > 1)]

if len(init_phase) > 0:
    popt, pcov = curve_fit(exp_growth, init_phase['time'], init_phase['I'], p0=(1, 0.1))
    a_fit, b_fit = popt
    doubling_time = np.log(2) / b_fit
else:
    doubling_time = np.nan

# Save variables for return
init_phase_points = len(init_phase)


doubling_time, init_phase_points