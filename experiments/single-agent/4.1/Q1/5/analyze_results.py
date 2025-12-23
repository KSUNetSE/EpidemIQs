
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
