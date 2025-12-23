
import pandas as pd
import os
import numpy as np

# Load SIR simulation results
csv_path = os.path.join(os.getcwd(), 'output', 'results-11.csv')
data = pd.read_csv(csv_path)

# Metrics to extract: Epidemic Duration, Peak Infection Rate, Peak Time, Final Epidemic Size, Doubling Time
# 1. Epidemic Duration (time when infected fraction < 1 for last time)
t = data['time'].values
I = data['I'].values
S = data['S'].values
R = data['R'].values
N = S[0] + I[0] + R[0]

infected_frac = I / N
above_thr = infected_frac > 0.001
if np.any(above_thr):
    epidemic_end_time = t[above_thr][-1]
else:
    epidemic_end_time = t[-1]

# 2. Peak infection rate and peak time
peak_infected = np.max(infected_frac)
peak_time = t[np.argmax(infected_frac)]

# 3. Final epidemic size (fraction recovered at end)
final_size = R[-1] / N

# 4. Doubling time: time for infected to double early
first_two = np.where(I > I[0]*2)[0]
if len(first_two) > 0:
    doubling_time = t[first_two[0]] - t[0]
else:
    doubling_time = np.nan

metrics = {
    'epidemic_duration_days': float(epidemic_end_time),
    'peak_infected_fraction': float(peak_infected),
    'peak_time_days': float(peak_time),
    'final_epidemic_size_fraction': float(final_size),
    'doubling_time_days': float(doubling_time)
}

metrics