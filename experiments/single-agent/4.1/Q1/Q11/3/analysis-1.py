
# Load simulation results for both networks, extract metrics. Focus on epidemic size, peak prevalence, peak time, duration.
import pandas as pd
import numpy as np

csv_hom = 'output/results-1-1.csv'
csv_het = 'output/results-1-2.csv'

results = {}
for label, csv_path in [('homogeneous', csv_hom), ('heterogeneous', csv_het)]:
    df = pd.read_csv(csv_path)
    N = df[['S', 'E', 'I', 'R']].iloc[0].sum()
    # Peak infected, time of peak
    peak_I = df['I'].max()
    peak_time = df['time'][df['I'].idxmax()]
    # Final epidemic size (total ever infected: ending R)
    final_size = df['R'].iloc[-1]
    # Duration (when I < 1 after main epidemic, after 5 time units past peak)
    duration = df[df['I'] < 1]['time']
    duration = (duration[duration > peak_time+5].min() if not duration[duration > peak_time+5].empty else df['time'].iloc[-1])
    results[label] = {
        'peak_infected': int(round(peak_I)),
        'peak_time': float(peak_time),
        'final_epidemic_size': int(round(final_size)),
        'duration': float(duration)
    }
results
