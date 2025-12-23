
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

data = pd.read_csv(os.path.join(os.getcwd(), 'output', 'results-11.csv'))

# Metrics for SIR (chain breaks due to lack of infectives)
duration = data['time'].iloc[-1]  # simulation time
final_s = data['S'].iloc[-1]
final_i = data['I'].iloc[-1]
final_r = data['R'].iloc[-1]
peak_I = data['I'].max()
peak_time = data['time'][data['I'].idxmax()]
final_size = final_r
peak_prop = peak_I / (data['S'][0] + data['I'][0] + data['R'][0])

doubling_times = []
I = data['I'].values
T = data['time'].values
for t in range(1, len(I)):
    if I[t] >= 2 * I[0]:
        doubling_times.append(T[t])
        break
if doubling_times:
    doubling = doubling_times[0]
else:
    doubling = np.nan

metrics = {
    'Epidemic Duration': float(duration),
    'Final Susceptibles': int(final_s),
    'Final Infectives': int(final_i),
    'Final Removed': int(final_r),
    'Final Epidemic Size (R)': int(final_size),
    'Peak Infectives': int(peak_I),
    'Peak Time': float(peak_time),
    'Peak Infection Proportion': float(peak_prop),
    'Initial Doubling Time': float(doubling)
}
metrics
