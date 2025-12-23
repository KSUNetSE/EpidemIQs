
import pandas as pd
import numpy as np
import os
# === LOAD SIMULATION RESULTS ===
results_path_BA = os.path.join(os.getcwd(), 'output', 'results-12.csv')
df_BA = pd.read_csv(results_path_BA)
def analyze_timeseries(df, pop_size=1000):
    met = {}
    met['final_epidemic_size'] = int(df['R'].iloc[-1])
    met['peak_infection'] = int(df['I'].max())
    t_peak = df['time'][df['I'].idxmax()]
    met['time_peak_infection'] = float(t_peak)
    I_nonzero = df.query('I>0')
    met['epidemic_duration'] = float(I_nonzero['time'].iloc[-1] - I_nonzero['time'].iloc[0])
    # Doubling time: time for I to double from 10 to 20 (if possible)
    I_10 = df.query('I>=10')
    t10 = I_10['time'].iloc[0] if not I_10.empty else np.nan
    I_20 = df.query('I>=20')
    t20 = I_20['time'].iloc[0] if not I_20.empty else np.nan
    met['doubling_time'] = float(t20-t10) if (not np.isnan(t10) and not np.isnan(t20)) else np.nan
    return met
metrics_BA = analyze_timeseries(df_BA)
return_vars=['metrics_BA']
