
# For completeness, summarize the main simulation CSV for scenario 1 (temporal, beta=1.55763)
import pandas as pd
import os
import numpy as np

data = pd.read_csv(os.path.join(os.getcwd(), 'output', 'results-01.csv'))
I_mean = data['I_mean'].values
R_mean = data['R_mean'].values
peak_I = np.argmax(I_mean)
t_peak = data['t'][peak_I]
final_size = R_mean[-1]
sum_stats = {'peak_I': float(I_mean[peak_I]), 't_peak': int(t_peak), 'final_size': float(final_size)}