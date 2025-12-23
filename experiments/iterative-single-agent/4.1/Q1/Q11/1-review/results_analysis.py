
import pandas as pd
import numpy as np
import os
# Analysis code to extract metrics from CSV results
results = {}
for i, j, name in [(1,1,'er'), (1,2,'ba')]:
    p = os.path.join(os.getcwd(),'output',f'results-{i}{j}.csv')
    df = pd.read_csv(p)
    # Extract metrics: epidemic peak (I max), peak time, final epidemic size (R[-1]), duration (last time infection >0)
    epi_peak = df['I'].max()
    epi_time = df.loc[df['I'].idxmax(),'time']
    final_size = df['R'].iloc[-1]
    duration = df.loc[df['I']>0,'time'].iloc[-1] if (df['I']>0).any() else np.nan
    time_to_extinct = df.loc[df['I']==0,'time'].iloc[-1] if (df['I']==0).any() else np.nan
    results[name] = dict(
        peak_I=float(epi_peak),
        peak_time=float(epi_time),
        final_size=float(final_size),
        duration=float(duration),
        time_to_extinct=float(time_to_extinct))
results