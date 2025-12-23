
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
# Load simulation results
results_path = os.path.join(os.getcwd(), 'output', 'results-11.csv')
data = pd.read_csv(results_path)
# Calculate metrics
peak_infection = np.max(data['I'])/1000  # fraction of population
peak_time = data['time'][np.argmax(data['I'])]
final_epidemic_size = data['R'].iloc[-1]/1000  # fraction of population
threshold_cross = data['time'][np.where(data['I'] > 10)[0][0]] if np.any(data['I'] > 10) else None
# Epidemic duration = time until I < 1 (effectively over)
end_idx = np.where(data['I'] < 1)[0]
epidemic_duration = data['time'].iloc[end_idx[0]] if len(end_idx) else data['time'].iloc[-1]
# Plot result for analysis report
data.plot(x='time', y=['S','I','R'])
plt.title('SIR Simulation Results')
plt.xlabel('Time (days)')
plt.ylabel('Individuals')
plt.savefig(os.path.join(os.getcwd(), 'output', 'SIR_timeseries_analysis.png'))
plt.close()
metrics = {
    'peak_infection_fraction': float(peak_infection),
    'peak_time': float(peak_time),
    'final_epidemic_size_fraction': float(final_epidemic_size),
    'epidemic_duration': float(epidemic_duration),
    'threshold_time_Igt10': float(threshold_cross) if threshold_cross is not None else None
}
metrics, os.path.join(os.getcwd(), 'output', 'SIR_timeseries_analysis.png')
