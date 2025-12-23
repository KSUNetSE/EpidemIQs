
# Re-calculate static network summary statistics using correct columns
import pandas as pd
import os
import numpy as np

data = pd.read_csv(os.path.join(os.getcwd(), 'output', 'results-21.csv'))
I = data['I'].values
R = data['R'].values
peak_I = np.argmax(I)
t_peak = int(data['time'][peak_I])
final_size = float(R[-1])
final_size_frac = final_size / 10000  # population N
summary_stats = {'peak_I': int(peak_I),
                 't_peak': int(t_peak),
                 'final_size_abs': float(final_size),
                 'final_size_frac': float(final_size_frac)}