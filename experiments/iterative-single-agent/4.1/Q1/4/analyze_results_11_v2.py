
import pandas as pd
results = pd.read_csv('output/results-11.csv')
# Locate peak infection row numerically
peak_I = results['I'].max()
peak_idx = results['I'].idxmax()
peak_time = results.loc[peak_idx, 'time']
# Final epidemic size (R at last time point)
final_R = results['R'].iloc[-1]
# Epidemic duration: difference between first and last nonzero I
infected = results[results['I']>0]
epidemic_duration = infected['time'].iloc[-1] - infected['time'].iloc[0]
metrics = {
    'Peak Infection': float(peak_I),
    'Final Epidemic Size (R)': float(final_R),
    'Epidemic Duration (days)': float(epidemic_duration),
    'Peak Time (days)': float(peak_time),
    'First infection time': float(infected['time'].iloc[0]),
    'Last infection time': float(infected['time'].iloc[-1])
}
metrics