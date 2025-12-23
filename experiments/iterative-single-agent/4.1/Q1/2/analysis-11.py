
import pandas as pd
import numpy as np
import os

# Load simulation results
df = pd.read_csv(os.path.join(os.getcwd(), 'output', 'results-11.csv'))

def get_peak_infection_metrics(df):
    # Find peak infected and its time
    peak_I = df['I'].max()
    peak_time = df.loc[df['I'].idxmax(), 'time']
    # Final epidemic size (# recovered at final time)
    final_R = df['R'].iloc[-1]
    # Doubling time: time it takes for I to go from 10->20 (if possible)
    doubling_time = np.nan
    i_start = df['I'].ge(10).idxmax()
    i2 = df['I'].ge(20).idxmax() if (df['I']>=20).any() else None
    if i2 and i2 > i_start:
        doubling_time = df['time'][i2] - df['time'][i_start]
    # Epidemic duration: time from first I>10 to when I<1
    end_idx = df['I'].le(1).to_numpy().nonzero()[0]
    if len(end_idx) > 0 and end_idx[-1] > i_start:
        duration = df['time'].iloc[end_idx[-1]] - df['time'].iloc[i_start]
    else:
        duration = np.nan
    return {
        'Peak Infection': int(peak_I),
        'Peak Time': float(peak_time),
        'Final Epidemic Size (R_inf)': int(final_R),
        'Doubling Time': float(doubling_time) if not np.isnan(doubling_time) else 'N/A',
        'Epidemic Duration': float(duration) if not np.isnan(duration) else 'N/A',
    }

metrics = get_peak_infection_metrics(df)

# Summarize for report
summary = f"""
Peak infected number: {metrics['Peak Infection']} at t={metrics['Peak Time']:.1f}\nFinal epidemic size (total recovered): {metrics['Final Epidemic Size (R_inf)']}\nDoubling time: {metrics['Doubling Time']}\nEpidemic duration: {metrics['Epidemic Duration']} time units\n"""
(summary, metrics)