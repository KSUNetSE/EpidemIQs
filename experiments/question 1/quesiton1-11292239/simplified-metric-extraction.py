
import numpy as np

# Population size
N = 2000

# Normalize S, E, I, R
for col in ['S', 'E', 'I', 'R']:
    data[col+'_norm'] = data[col] / N

# Epidemic duration: last time where I >= 1 (I_norm >= 1/N = 0.0005)
threshold = 1 / N
epidemic_duration = data.loc[data['I_norm'] >= threshold, 'time'].max()  # time units same as in data

# Peak infection prevalence and timing
peak_idx = data['I_norm'].idxmax()
peak_infection_prevalence = data.loc[peak_idx, 'I_norm']
peak_infection_time = data.loc[peak_idx, 'time']

# Final epidemic size: last value of R normalized
final_epidemic_size = data.iloc[-1]['R_norm']

# Early phase definition for doubling time
early_cutoff = 1  # time units
early_phase_data = data[data['time'] <= early_cutoff]
early_I = early_phase_data['I_norm']
early_time = early_phase_data['time']

# Only use strictly positive I values for log analysis
valid_mask = early_I > 0
early_I = early_I[valid_mask]
early_time = early_time[valid_mask]

if len(early_I) > 1:
    coeffs = np.polyfit(early_time, np.log(early_I), 1)  # ln(I) = slope * t + intercept
    growth_rate = coeffs[0]
    initial_doubling_time = np.log(2) / growth_rate if growth_rate > 0 else np.inf
else:
    initial_doubling_time = None

(e, p, t, f, d) = (epidemic_duration, peak_infection_prevalence, peak_infection_time, final_epidemic_size, initial_doubling_time)