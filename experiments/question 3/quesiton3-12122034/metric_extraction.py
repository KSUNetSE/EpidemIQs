
import numpy as np

# Extract relevant data series
S = data['SIR_S'].values
I = data['SIR_I'].values
R = data['SIR_R'].values

time = data['time'].values

# Final epidemic size (mean approx from final R)
R_final = R[-1]

# Peak I and time to peak
peak_I = np.max(I)
peak_time = time[np.argmax(I)]

# Epidemic duration: time till infected drops below 1 (can be taken as end of epidemic)
infected_below_threshold_indices = np.where(I < 1)[0]
epidemic_duration = time[infected_below_threshold_indices[0]] if infected_below_threshold_indices.size > 0 else time[-1]

# Early fade-out probability estimation: not available from aggregate, use lower CI at final time point on R to guess if final recovered <10
R_final_lower = data['SIR_R_90ci_lower'].values[-1]
early_fade_out = R_final_lower < 10

# Approximate doubling time in early phase using infected data
# Select early time window where I is growing exponentially
early_time_mask = (time <= 1.0)
early_times = time[early_time_mask]
early_I = I[early_time_mask]

# Fit exponential model to early infected: I(t) = I0*exp(rt), doubling time = ln(2)/r
from scipy.optimize import curve_fit

def exp_func(t, I0, r):
    return I0 * np.exp(r * t)

try:
    popt, _ = curve_fit(exp_func, early_times, early_I, p0=(I[0], 1))
    r_est = popt[1]
    doubling_time = np.log(2) / r_est if r_est > 0 else np.nan
except Exception:
    doubling_time = np.nan

# Produce results
results = {
    'final_epidemic_size_mean': R_final,              # mean approx. from final R (unit: individuals)
    'peak_infected': peak_I,                          # max infected (unit: individuals)
    'time_to_peak': peak_time,                        # time of peak infected (unit: time units used in data)
    'epidemic_duration': epidemic_duration,          # duration till infected near zero (unit: time units)
    'early_fade_out_prob_est': early_fade_out,       # boolean estimate if early fade out likely
    'doubling_time_early_phase': doubling_time       # doubling time in early phase (unit: time units)
}

results