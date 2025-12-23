
import pandas as pd
import numpy as np
import os

# Load SIR simulation results
df = pd.read_csv(os.path.join(os.getcwd(), 'output', 'results-11.csv'))
# Metrics to extract:
# 1. Epidemic duration: time until I < 1 (since total population 1000, when I~0, i.e., end of epidemic)
I = df['I'].values
S = df['S'].values
R = df['R'].values
time = df['time'].values
N = S[0] + I[0] + R[0]

# 1. Epidemic duration
duration = time[np.argmax(I<1)] if np.any(I<1) else time[-1]
# 2. Peak infection rate and peak size
peak_I = np.max(I)
peak_time = time[np.argmax(I)]
# 3. Final epidemic size (final number recovered)
final_size = R[-1]
# 4. Doubling time: time to go from I[0] to 2*I[0] (if possible)
dbl_idx = np.where(I >= 2*I[0])[0]
doubling_time = time[dbl_idx[0]]-time[0] if dbl_idx.size > 0 else np.nan
# 5. Proportion infected
prop_infected = final_size/N

metrics = {
  'Epidemic Duration (days)': float(duration),
  'Peak Prevalence': float(peak_I),
  'Peak Time (days)': float(peak_time),
  'Final Epidemic Size': float(final_size),
  'Doubling Time (days)': float(doubling_time),
  'Proportion Infected (final)': float(prop_infected)
}
metrics