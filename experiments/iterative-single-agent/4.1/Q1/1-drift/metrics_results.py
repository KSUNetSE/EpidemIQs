
# Quantitative analysis: Read both CSVs, extract epidemic metrics: peak I (max), final epidemic size (R at end), peak timing, duration
import pandas as pd
import numpy as np

results = {}
for idx, (name, csvpath) in enumerate(zip(['Homogeneous (ER)', 'Heterogeneous (BA)'], ['output/results-11.csv', 'output/results-12.csv'])):
    df = pd.read_csv(csvpath)
    # Peak infection: max I
    peak_I = df['I'].max()
    t_peak = df.loc[df['I'].idxmax(), 'time']
    # Final size: R at end
    final_size = df['R'].iloc[-1]
    # Epidemic duration: last time when E+I > 0
    infected_periods = (df['E']+df['I']) > 0
    epidemic_duration = df['time'][infected_periods].max() if infected_periods.any() else 0
    results[name] = {'Peak_I': int(peak_I), 'Time_Peak': float(t_peak), 'Final_Size': int(final_size), 'Epidemic_Duration': float(epidemic_duration)}
results