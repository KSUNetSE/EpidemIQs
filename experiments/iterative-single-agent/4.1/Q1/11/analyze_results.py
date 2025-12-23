
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
