
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the simulation CSV
data = pd.read_csv('output/results-11.csv')

# Extract basic metrics
n = 500
t_last = data['time'].iloc[-1]
I1_final = data['I1'].iloc[-1]
I2_final = data['I2'].iloc[-1]
S_final = data['S'].iloc[-1]

# Check for coexistence: both I1, I2 > 0 (as fraction of n)
coexist = (I1_final > 2) and (I2_final > 2)
# Final sizes
frac_I1 = I1_final/n
frac_I2 = I2_final/n
frac_S = S_final/n

# Peak infection
I1_peak = data['I1'].max()
I2_peak = data['I2'].max()
I1_peak_time = data['time'][data['I1'].idxmax()]
I2_peak_time = data['time'][data['I2'].idxmax()]

# Epidemic duration (time infected population remains above 2 individuals)
def epidemic_duration(col, threshold=2):
    above = (data[col] > threshold)
    if above.any():
        t_start = data['time'][above.idxmax()]
        t_end = data['time'][len(data)-above[::-1].idxmax()-1]  # first-from-end where above
        return float(t_end - t_start)
    else:
        return 0
T_I1 = epidemic_duration('I1')
T_I2 = epidemic_duration('I2')

# Plot - check what the competition looks like
data.set_index('time')[['S','I1','I2']].plot()
plt.title('Dynamics of Competitive SIS: Multiplex')
plt.ylabel('Count')
plt.xlabel('Time')
plt.savefig('output/analysis_plot_11.png')

metrics = {
    'coexist': coexist,
    'final_I1_frac': frac_I1,
    'final_I2_frac': frac_I2,
    'final_S_frac': frac_S,
    'I1_peak': I1_peak,
    'I2_peak': I2_peak,
    'I1_peak_time': float(I1_peak_time),
    'I2_peak_time': float(I2_peak_time),
    'T_I1': T_I1,
    'T_I2': T_I2
}
metrics