
# 4) Final epidemic size and attack rate
final_R = R[-1]
attack_rate = final_R / N * 100

# 5) Doubling time
# We define doubling time during initial exponential growth phase
# Find time points where infection is growing
I_growth_phase = I[I > 0]
time_growth_phase = time[I > 0]

# Select the initial period until the peak for fitting exponential growth
growth_time = time_growth_phase[time_growth_phase <= peak_I_time]
growth_I = I[time <= peak_I_time]

# Initial I values must be >0 for log transform
growth_I = growth_I[growth_I > 0]
growth_time = growth_time[:len(growth_I)]

import numpy as np
from scipy.optimize import curve_fit

def exponential_func(t, a, b):
    return a * np.exp(b * t)

# Fit exponential curve to initial growth phase
params, cov = curve_fit(exponential_func, growth_time, growth_I, p0=[growth_I[0], 0.1])

# Doubling time is ln(2)/b
b = params[1]
doubling_time = np.log(2) / b

final_R, attack_rate, doubling_time