
import pandas as pd
import numpy as np

data = pd.read_csv('output/results-11.csv')

# Compute key metrics for the SIR simulation
peak_infected = data['I'].max()
peak_time = data['time'][data['I'].idxmax()]
final_epidemic_size = data['R'].iloc[-1]
epidemic_duration = data['time'][np.where(data['I'] < 1)[0][0]] if np.any(data['I'] < 1) else data['time'].iloc[-1]
doubling_time = None

# Doubling time estimation around epidemic growth phase
# Find first 2x increase in I
I_ = data['I']
t_ = data['time']
idx0 = np.argmax(I_ > 1)
I0 = I_[idx0]
idx_dbl = np.argmax(I_ > 2*I0)
if idx0 < idx_dbl:
    doubling_time = t_[idx_dbl] - t_[idx0]
else:
    doubling_time = None

metrics = {
    'peak_infected': int(peak_infected),
    'peak_time': float(peak_time),
    'final_epidemic_size': int(final_epidemic_size),
    'epidemic_duration': float(epidemic_duration),
    'doubling_time': float(doubling_time) if doubling_time is not None else None
}

metrics
