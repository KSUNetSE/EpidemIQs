
# Chain-of-Thought: Load and compare both results for key metrics.
# Extract: Epidemic duration (time for I to drop to <1% N), Peak I, Final R, Peak time, for both network types.
import pandas as pd
import numpy as np
import os

res_static = pd.read_csv(os.path.join(os.getcwd(),'output','results-1-1.csv'))
res_temp = pd.read_csv(os.path.join(os.getcwd(),'output','results-1-2.csv'))
N = 1000
metrics = {}
def extract_metrics(df):
    peak_I = df['I'].max()
    peak_time = df['I'].idxmax()
    final_R = df['R'].iloc[-1]
    # Epidemic duration = last time when I > 0.01*N
    thresh = 0.01*N
    active = (df['I'] > thresh)
    if np.any(active):
        t_end = np.where(active)[0][-1]
    else:
        t_end = np.nan
    return {'peak_I': int(peak_I), 'peak_time': int(peak_time), 'final_R': int(final_R), 'duration': int(t_end) if not np.isnan(t_end) else None}
metrics['static'] = extract_metrics(res_static)
metrics['temporal'] = extract_metrics(res_temp)
metrics
