
import pandas as pd
import numpy as np
import os

def extract_metrics(result_csv_path):
    df = pd.read_csv(result_csv_path)
    N = df[['S','E','I','R']].sum(axis=1).iloc[0]  # total pop (should be ~1000)
    # 1. Peak I count and its time
    peak_I = df['I'].max()
    peak_time = df.loc[df['I'].idxmax(), 'time']
    # 2. Final epidemic size (total R at end)
    final_size = df['R'].iloc[-1]
    # 3. Epidemic duration (time from >1 to <1 I)
    active = df['I'] > 1
    if active.any():
        duration = df['time'][active].max() - df['time'][active].min()
    else:
        duration = 0
    # 4. Doubling time around initial outbreak
    I_series = df['I'].values
    t_series = df['time'].values
    start_idx = np.argmax(I_series > 5)
    end_idx = np.argmax(I_series > 2*I_series[start_idx]) if (2*I_series[start_idx] < I_series.max()) else start_idx+2
    doubling_time = t_series[end_idx] - t_series[start_idx] if end_idx > start_idx else np.nan
    return {
        'peak_I': float(peak_I),
        'peak_time': float(peak_time),
        'final_size': float(final_size),
        'duration': float(duration),
        'doubling_time': float(doubling_time)
    }

metrics_er = extract_metrics(os.path.join(os.getcwd(),'output','results-1-1.csv'))
metrics_ba = extract_metrics(os.path.join(os.getcwd(),'output','results-1-2.csv'))

metrics = {'er': metrics_er, 'ba': metrics_ba}

import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import os

df = pd.read_csv(os.path.join(os.getcwd(), 'output', 'results-11.csv'))
# Epidemic metrics
# 1. Epidemic duration: last time I > 0
active_idx = np.where(df['I'] > 0)[0]
epidemic_duration = df['time'].iloc[active_idx[-1]] - df['time'].iloc[active_idx[0]] if len(active_idx) > 0 else 0
# 2. Peak infection
peak_infection = df['I'].max()
peak_time = df.loc[df['I'].idxmax(), 'time']
# 3. Final epidemic size (total removed)
final_R = df['R'].iloc[-1]
# 4. Doubling time (initial phase: between 10% and 20% of max)
max_I = peak_infection
if max_I > 20:  # epidemic grows enough
    sel = (df['I'] > 0.1*max_I) & (df['I'] < 0.2*max_I)
    t_dt = df.loc[sel, 'time']
    I_dt = df.loc[sel, 'I']
    doubling_time = (t_dt.iloc[-1] - t_dt.iloc[0]) / np.log2(I_dt.iloc[-1]/I_dt.iloc[0]) if (len(t_dt)>1 and I_dt.iloc[0]>0) else np.nan
else:
    doubling_time = np.nan
# Prepare metrics to save
metrics = dict(
    epidemic_duration=float(epidemic_duration),
    peak_infection=int(peak_infection),
    peak_time=float(peak_time),
    final_epidemic_size=int(final_R),
    doubling_time=float(doubling_time)
)
metrics
