
import pandas as pd

df1 = pd.read_csv('output/results-11.csv')
df2 = pd.read_csv('output/results-12.csv')

# Metrics computed: final epidemic size, peak infection, time to peak, final S, final R
metrics = {}
for run, df in zip(['random_vax','deg10_vax'],[df1,df2]):
    final_size = df['R'].max() # assumes everyone who was infected is recovered at end
    peak_I = df['I'].max()
    t_peak = df['I'].idxmax() if df['I'].max() > 0 else None
    final_S = df['S'].iloc[-1]
    metrics[run] = {
        'final_epidemic_size': int(final_size),
        'peak_infection': int(peak_I),
        'time_to_peak': int(t_peak) if t_peak is not None else None,
        'final_S': int(final_S)
    }
metrics
