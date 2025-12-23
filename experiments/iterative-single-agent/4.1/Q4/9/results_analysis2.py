
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load simulation results
results_path = 'output/results-12.csv'
data = pd.read_csv(results_path)
time = data['time']
S = data['S']
I = data['I']
R = data['R']

# Metrics extraction:
peak_infected = I.max()
peak_time = time[I.idxmax()]
final_size = R.iloc[-1]
epidemic_duration = time[R > 0].iloc[-1] if np.any(R > 0) else time.iloc[-1]

def doubling_time(df):
    try:
        start = df['I'].values[0]
        double_val = start * 2
        above = np.where(df['I'].values >= double_val)[0]
        return df['time'].iloc[above[0]] if above.size > 0 else np.nan
    except Exception:
        return np.nan

dbl_time = doubling_time(data)

# Compose output (metrics)
metrics = {
    'Peak Infected': peak_infected,
    'Peak Time (days)': peak_time,
    'Final Epidemic Size (R at end)': final_size,
    'Epidemic Duration (days)': epidemic_duration,
    'Doubling Time (days)': dbl_time
}
metrics