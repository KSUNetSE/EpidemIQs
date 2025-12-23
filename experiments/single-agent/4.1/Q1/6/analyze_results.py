
# Loading simulation results for analysis
import pandas as pd
import os
import numpy as np

results_path = os.path.join(os.getcwd(), 'output', 'results-11.csv')
data = pd.read_csv(results_path)

# Calculate metrics: final epidemic size, peak infection, peak time, duration
final_size = int(data['R'].iloc[-1])
peak_infection = int(np.max(data['I']))
peak_time = float(data['time'][np.argmax(data['I'])])
duration = float(data['time'].iloc[-1])

metrics = {
    'final_epidemic_size': final_size,
    'peak_infection': peak_infection,
    'peak_time': peak_time,
    'epidemic_duration': duration
}

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Load the simulation results
results_path = os.path.join(os.getcwd(), 'output', 'results-11.csv')
data = pd.read_csv(results_path)

# Calculate epidemic statistics
pop_size = 1000

# Final epidemic size (fraction of Recovered at last timepoint)
final_epidemic_size = data['R'].iloc[-1] / pop_size

# Peak infection prevalence (max(I))
peak_prevalence = data['I'].max() / pop_size

# Peak time
peak_time = data.loc[data['I'].idxmax(), 'time']

# Total duration (when I falls below 1)
i_below_1_idx = np.where(data['I'] < 1)[0]
if len(i_below_1_idx) == 0:
    epidemic_duration = data['time'].iloc[-1]
else:
    epidemic_duration = data['time'].iloc[i_below_1_idx[0]]

# Doubling time (early exponential phase)
def doubling_time(arr, times, window=(2,20)):
    from scipy.stats import linregress
    idx = np.arange(window[0], window[1])
    inc = arr[idx]
    tvals = times[idx]
    inc = np.log(inc.clip(min=1))  # avoid log(0)
    slope, *_ = linregress(tvals, inc)
    return np.log(2)/slope if slope > 0 else np.nan

dt = doubling_time(data['I'].values, data['time'].values)

# Plot summary in case of appendix usage
plt.figure()
plt.plot(data['time'], data['S']/pop_size, label='S')
plt.plot(data['time'], data['I']/pop_size, label='I')
plt.plot(data['time'], data['R']/pop_size, label='R')
plt.xlabel('Time (days)')
plt.ylabel('Fraction of Population')
plt.legend()
plt.title('Epidemic Dynamics - SIR on ER network')
plt.tight_layout()
plt.savefig(os.path.join(os.getcwd(), 'output', 'epi_summary.png'))
plt.close()

# Output metrics
{
    'final_epidemic_size': final_epidemic_size,
    'peak_prevalence': peak_prevalence,
    'peak_time': peak_time,
    'epidemic_duration': epidemic_duration,
    'doubling_time': dt,
}