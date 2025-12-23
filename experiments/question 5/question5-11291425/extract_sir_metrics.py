
N = 10000

# Normalize the columns by N
data['S_frac'] = data['S'] / N
data['I_frac'] = data['I'] / N
data['R_frac'] = data['R'] / N

# 1. Final size (max R/N)
final_size = data['R_frac'].max()

# 2. Peak infection prevalence (max I/N)
peak_infection_prevalence = data['I_frac'].max()

# 3. Time of peak infection prevalence
time_of_peak = data.loc[data['I_frac'].idxmax(), 'time']

final_size, peak_infection_prevalence, time_of_peak
# 4. Epidemic duration: find the last time point when I is above a small threshold (e.g. 0.001 of population)
thresh = 0.001
infected_above_thresh = data[data['I_frac'] > thresh]
epidemic_duration = infected_above_thresh['time'].iloc[-1] - infected_above_thresh['time'].iloc[0]

# 5. Doubling time estimation during early phase (e.g. first 10% of data time, when I is growing)
import numpy as np
from scipy.optimize import curve_fit

def exp_growth(t, a, b):
    return a * np.exp(b*t)

# Filter early phase data where I_frac is small and increasing
early_phase = data[data['time'] <= data['time'].quantile(0.1)]
early_phase_increasing = early_phase[(early_phase['I_frac'] > 0) & (early_phase['I_frac'].diff() > 0)]

# Use only positive values and times for fitting
times = early_phase_increasing['time'].values
infections = early_phase_increasing['I_frac'].values

# Fit exponential curve to early phase infection fraction
doubling_time = None
if len(times) > 2:
    try:
        popt, pcov = curve_fit(exp_growth, times, infections, p0=[infections[0], 1])
        growth_rate = popt[1]  # b parameter in exp(b*t)
        doubling_time = np.log(2) / growth_rate if growth_rate > 0 else None
    except Exception as e:
        doubling_time = None

epidemic_duration, doubling_time