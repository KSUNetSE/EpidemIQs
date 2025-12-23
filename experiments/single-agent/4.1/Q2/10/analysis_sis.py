
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

data = pd.read_csv(os.path.join(os.getcwd(), 'output', 'results-12.csv'))

# Metrics for SIS (should maintain endemic infection if R0 > 1)
duration = data['time'].iloc[-1]  # simulation time
final_s = data['S'].iloc[-1]
final_i = data['I'].iloc[-1]
peak_I = data['I'].max()
peak_time = data['time'][data['I'].idxmax()]
endemic_inf = np.mean(data['I'][-10:])  # average last 10 time points as endemic level
peak_prop = peak_I / (data['S'][0] + data['I'][0])

I = data['I'].values
T = data['time'].values

doubling_times = []
for t in range(1, len(I)):
    if I[t] >= 2 * I[0]:
        doubling_times.append(T[t])
        break
if doubling_times:
    doubling = doubling_times[0]
else:
    doubling = np.nan

metrics = {
    'Simulation Duration': float(duration),
    'Final Susceptibles': int(final_s),
    'Final Infectives': int(final_i),
    'Peak Infectives': int(peak_I),
    'Peak Time': float(peak_time),
    'Endemic Infective Level': float(endemic_inf),
    'Peak Infection Proportion': float(peak_prop),
    'Initial Doubling Time': float(doubling)
}
metrics
