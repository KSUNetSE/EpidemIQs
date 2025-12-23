
import pandas as pd
import numpy as np
# Load simulation results
results = pd.read_csv('/Users/hosseinsamaei/phd/gemf_llm/output/results-11.csv')

# Metrics:
# Epidemic Duration (last time more than 1% are infected)
I = results['I']
time = results['time']

peak_infected = I.max()
peak_time = time[I.argmax()]

epidemic_size_final = results['R'].iloc[-1]

# Find time when I < 1% after initial outbreak
threshold = 0.01 * (I + results['S'] + results['R']).iloc[0]
above_thr = np.where(I > threshold)[0]
if len(above_thr) > 0:
    epidemic_duration = time[above_thr[-1]]
else:
    epidemic_duration = 0

doubling_time = None
initial_idx = np.where(I > 0)[0][0]
if I[initial_idx] > 0:
    # Find when I doubles for the first time
    target = 2 * I[initial_idx]
    greater = np.where(I >= target)[0]
    if len(greater) > 0:
        doubling_time = time[greater[0]] - time[initial_idx]

metrics = {
    'peak_infected': int(peak_infected),
    'peak_time': float(peak_time),
    'final_epidemic_size': int(epidemic_size_final),
    'epidemic_duration': float(epidemic_duration),
    'doubling_time': float(doubling_time) if doubling_time is not None else None,
    'final_fraction_infected': float(epidemic_size_final) / (results['S'][0] + results['I'][0] + results['R'][0])
}
metrics