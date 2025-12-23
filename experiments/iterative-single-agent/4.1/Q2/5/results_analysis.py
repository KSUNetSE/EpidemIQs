
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

import pandas as pd
import numpy as np
# Load results csv
csv_path = '/Users/hosseinsamaei/phd/gemf_llm/output/results-11.csv'
df = pd.read_csv(csv_path)
# Find peak infection value and its time
i_peak = df['I'].max()
t_peak = df['time'][df['I'].idxmax()]
# Final epidemic size = total recovered at final time
final_recovered = df['R'].iloc[-1]
# Doubling time (approximate early stage growth rate log)
early_infected = df['I'].loc[5:15]
early_time = df['time'].loc[5:15]
if (early_infected > 0).all() and len(early_infected) > 5:
    from scipy.stats import linregress
    slope, _, _, _, _ = linregress(early_time, np.log(early_infected + 1e-6))
    doubling_time = np.log(2)/slope if slope > 0 else np.nan
else:
    doubling_time = np.nan
# Epidemic duration (when I last falls below 1)
idxs = np.where(df['I'] < 1)[0]
epi_duration = df['time'][idxs[0]] if len(idxs) > 0 else df['time'].iloc[-1]
metrics = {
    'peak_infection': i_peak,
    'peak_time': t_peak,
    'final_epidemic_size': final_recovered,
    'epidemic_duration': epi_duration,
    'doubling_time':float(doubling_time),
}
metrics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Load results
data_path = os.path.join(os.getcwd(), 'output', 'results-11.csv')
data = pd.read_csv(data_path)

# Extract metrics
N = 1000
final_size = (N - data['S'].iloc[-1]) / N  # proportion ever infected
peak_I = data['I'].max() / N
peak_time = data['time'][data['I'].idxmax()]
epidemic_duration = data['time'][np.where(data['I'] > 1)[0].max()] - data['time'][np.where(data['I'] > 1)[0].min()]
doubling_time = np.nan
I_vals = data['I'].values
for i in range(len(I_vals)):
    if I_vals[i] >= 2 * I_vals[0]:
        doubling_time = data['time'][i] - data['time'][0]
        break
# Basic reproduction number
R_0 = 2.5  # Input parameter

# Plot final compartments for table
final_states = {comp: int(data[comp].iloc[-1]) for comp in ['S','I','R']}
metrics_table_path = os.path.join(os.getcwd(), 'output', 'metrics_table.png')
fig, ax = plt.subplots()
cell_text = [[k, v, f'{(v/N):.2%}'] for k,v in final_states.items()]
ax.axis('off')
table = ax.table(cellText=cell_text, colLabels=["Compartment", "Final Count", "Proportion"], cellLoc='center', loc='center')
plt.title("Final Population by Compartment")
plt.savefig(metrics_table_path)
plt.close()

# Return findings
results = {
    'N': N,
    'final_epidemic_size': final_size,
    'peak_infection': peak_I,
    'peak_time': peak_time,
    'epidemic_duration': epidemic_duration,
    'final_S': final_states['S'],
    'final_I': final_states['I'],
    'final_R': final_states['R'],
    'doubling_time':doubling_time,
    'R_0': R_0,
    'metrics_table_path': metrics_table_path
}
results