
import pandas as pd
import numpy as np
import os
metrics_tbl = []
for k,label in [('1-1','ER network'),('1-2','BA network'),('1-3','Deterministic ODE')]:
    csv_path = os.path.join(os.getcwd(), 'output', f'results-{k}.csv')
    df = pd.read_csv(csv_path)
    size = df['R'].iloc[-1]
    peak_I = df['I'].max()
    t_peak = df.loc[df['I'].idxmax(),'time']
    duration = df.loc[df['I']>1].shape[0]*(df['time'][1] - df['time'][0])
    metrics_tbl.append({'model':label, 'final_epidemic_size':size, 'peak_infectious':peak_I, 'time_to_peak':t_peak, 'epidemic_duration':duration})
pd.DataFrame(metrics_tbl).to_csv(os.path.join(os.getcwd(), 'output', 'summary_metrics.csv'), index=False)
metrics_tbl