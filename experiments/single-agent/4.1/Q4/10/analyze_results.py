
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
import pandas as pd
import os

data = pd.read_csv(os.path.join('output','results-12.csv'))
res_dict = {}
# Epidemic size: final R
final_R = data['R'].iloc[-1]
# Max I (peak infection)
peak_I = data['I'].max()
# Time to peak
peak_time = data.loc[data['I'].idxmax(),'time']
# Epidemic duration: last time when I>1
above_thresh = data.query('I>1')
epidemic_duration = above_thresh['time'].iloc[-1] if not above_thresh.empty else 0
# Doubling time: just use first doubling in I
first = data.query('I>1')
if not first.empty and len(first)>2:
    t1 = first['time'].iloc[0]
    i1 = first['I'].iloc[0]
    doubled = first.query('I>=@i1*2')
    doubling_time = doubled['time'].iloc[0] if not doubled.empty else None
else:
    doubling_time = None
# R0 (already set, report it)
R0 = 2.6
res_dict['final_R'] = int(final_R)
res_dict['peak_I'] = int(peak_I)
res_dict['peak_time'] = peak_time
res_dict['epidemic_duration'] = epidemic_duration
res_dict['doubling_time'] = doubling_time
res_dict['R0']=R0
res_dict
# Analysis: load simulation data and extract metrics (peak I, final R, duration, peak time, doubling time)
import pandas as pd
import numpy as np

csv_path = '/Users/hosseinsamaei/phd/gemf_llm/output/results-11.csv'
data = pd.read_csv(csv_path)

time = data['time'].values
S = data['S'].values
I = data['I'].values
R = data['R'].values

# Peak infection and when
peak_I = int(I.max())
peak_time = float(time[I.argmax()])
# Final epidemic size
final_R = int(R[-1])
# Duration (active infections >1)
active = np.where(I > 0)[0]
duration = float(time[active[-1]] - time[active[0]]) if len(active) > 0 else 0
# Doubling time around early phase
I_early = I[(I > 0) & (I < 100)]
time_early = time[(I > 0) & (I < 100)]
doubling_time = 0.0
if len(I_early) > 1:
    growth_rates = np.diff(np.log(I_early)) / np.diff(time_early)
    if len(growth_rates) > 0 and np.mean(growth_rates) > 0:
        doubling_time = np.log(2)/np.mean(growth_rates)

metrics = {
    'Peak_Infection': peak_I,
    'Peak_Time': peak_time,
    'Final_Removed': final_R,
    'Duration': duration,
    'Doubling_Time': doubling_time
}
metrics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Load results
results_path = os.path.join(os.getcwd(), 'output', 'results-11.csv')
data = pd.read_csv(results_path)

# Epidemic metrics
peak_I = data['I'].max()
peak_time = data.loc[data['I'].idxmax(), 'time']
final_R = data['R'].iloc[-1]
epidemic_duration = data[data['I'] > 0]['time'].iloc[-1] - data[data['I'] > 0]['time'].iloc[0]
initial_I = data['I'][0]
doubling_time = np.log(2) / ((np.log(data['I'][10]) - np.log(initial_I)) / (data['time'][10] - data['time'][0])) if data['I'][10] > 0 and initial_I > 0 else np.nan

# Plot metrics table
metrics = {
    'Peak Infection': int(peak_I),
    'Peak Time (days)': float(peak_time),
    'Final Epidemic Size (Recovered)': int(final_R),
    'Epidemic Duration (days)': float(epidemic_duration),
    'Doubling Time (days)': float(doubling_time)
}

# Output full metric summary for the report
metrics