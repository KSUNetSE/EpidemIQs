
import os, pandas as pd, numpy as np
out_dir = os.path.join(os.getcwd(), 'output')
paths = {'ER': os.path.join(out_dir, 'results-11.csv'), 'BA': os.path.join(out_dir, 'results-12.csv')}
metrics = {}
for name, p in paths.items():
    df = pd.read_csv(p)
    N = df[['S','E','I','R']].iloc[0].sum()
    peak_I = df['I'].max()
    peak_time = df['time'][df['I'].idxmax()]
    final_R = df['R'].iloc[-1]
    duration = df['time'][ (df['I']>0).to_numpy().nonzero()[0][-1] ]
    metrics[name] = {'peak_I': peak_I, 'peak_prop':peak_I/N, 'peak_time': peak_time, 'final_R': final_R, 'attack_rate':final_R/N, 'duration': duration}
print(metrics)