
import pandas as pd
import os
import numpy as np

# Load data
res1 = pd.read_csv(os.path.join('output','results-11.csv'))
resw = pd.read_csv(os.path.join('output','results-12.csv'))

# Helper: Find last time infected drops below threshold (>0.001)
def epidemic_duration(df):
    I_frac = df['I']/1000
    above_th = np.where(I_frac > 0.001)[0]
    if len(above_th) == 0:
        return 0
    idx_last = above_th[-1]
    return df['time'].iloc[idx_last]

def extract_metrics(df):
    I_frac = df['I']/1000
    R_frac = df['R']/1000
    peak_I = np.max(I_frac)
    time_peak = df['time'].iloc[np.argmax(I_frac)]
    final_size = R_frac.iloc[-1]
    epi_duration = epidemic_duration(df)
    return dict(peak_infect=peak_I, time_peak=time_peak, final_size=final_size, duration=epi_duration)

metrics_res1 = extract_metrics(res1)
metrics_resw = extract_metrics(resw)

metrics_res1, metrics_resw