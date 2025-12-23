
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
# Load simulation results
output_dir = os.path.join(os.getcwd(), 'output')
results_path = os.path.join(output_dir, 'results-11.csv')
data = pd.read_csv(results_path)

metrics = {}
# Epidemic Duration = time from first infection to near extinction (I < 1)
I = data['I'].values
time = data['time'].values
initial_infected = I[0]
max_infected = np.max(I)
peak_time = time[np.argmax(I)]
# The time when infection falls below 1 after the peak
threshold = 1
try:
    duration_ix = np.where((I < threshold) & (time > peak_time))[0][0]
    duration_time = time[duration_ix] - time[0]
except:
    duration_time = np.nan
metrics['Epidemic Duration'] = duration_time
metrics['Peak Infection Number'] = max_infected
metrics['Peak Time'] = peak_time
metrics['Final Epidemic Size (R)'] = data['R'].iloc[-1]
metrics['Total Outbreak Size'] = data['R'].iloc[-1]  # SIR, so total recovered equals total infected

# Doubling Time during exponential growth
growth_indices = np.where((I > initial_infected) & (I < max_infected/2))[0]
doubling_times = []
for i in growth_indices:
    for j in range(i+1, len(I)):
        if I[j] >= 2*I[i]:
            doubling_times.append(time[j] - time[i])
            break
metrics['Mean Doubling Time'] = np.mean(doubling_times) if doubling_times else np.nan

# Plot overlay for results with extracted metrics
plt.figure()
plt.plot(data['time'], data['S'], label='S')
plt.plot(data['time'], data['I'], label='I')
plt.plot(data['time'], data['R'], label='R')
plt.axvline(peak_time, color='k', linestyle='--', label='Peak Infection')
plt.title('SIR Simulation Results with Metrics')
plt.xlabel('Time')
plt.ylabel('Number of individuals')
plt.legend()
plt.grid(True)
plt.tight_layout()

metric_fig_path = os.path.join(output_dir, 'results-metrics-11.png')
plt.savefig(metric_fig_path)
plt.close()

# Report metrics and figure path
{'metrics': metrics, 'fig_path': metric_fig_path}