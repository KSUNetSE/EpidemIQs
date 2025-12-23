
import pandas as pd
import numpy as np
import os

output_dir = os.path.join(os.getcwd(), 'output')

data_agg = pd.read_csv(os.path.join(output_dir, 'results-1-1.csv'))
data_low = pd.read_csv(os.path.join(output_dir, 'results-1-2.csv'))

results = {}
for label, df in [('aggregated', data_agg), ('instantaneous', data_low)]:
    max_I = df['I'].max()
    t_peak = df['time'][df['I'].idxmax()]
    final_R = df['R'].iloc[-1]
    results[label] = {
        'peak_infection': int(max_I),
        'peak_time': float(t_peak),
        'final_epidemic_size': int(final_R)
    }

results['aggregated'], results['instantaneous']
