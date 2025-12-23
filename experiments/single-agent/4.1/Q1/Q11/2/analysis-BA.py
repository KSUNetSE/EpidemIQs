
# Analysis for BA (Scale-Free)
import pandas as pd
import numpy as np
results = pd.read_csv('output/results-11.csv')

epidemic_duration = results['time'].iloc[-1]
peak_infected = results['I'].max()
peak_time = results['time'][results['I'].idxmax()]
final_epidemic_size = results['R'].iloc[-1]
final_S = results['S'].iloc[-1]

metrics_BA = {
    'epidemic_duration': epidemic_duration,
    'peak_infected': int(peak_infected),
    'peak_time': float(peak_time),
    'final_epidemic_size': int(final_epidemic_size),
    'final_susceptible': int(final_S),
    'network_type': 'BA'
}
metrics_BA