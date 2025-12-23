
# Load and analyze stochastic simulation results
import pandas as pd
import numpy as np

sim_er = pd.read_csv('output/results-11.csv')
sim_ba = pd.read_csv('output/results-12.csv')
sim_det = pd.read_csv('output/results-13.csv')

# Metrics: Final epidemic size, peak I (absolute and time), peak new daily infections
# Final size: total R at end
final_R_er = sim_er['R'].iloc[-1]
final_R_ba = sim_ba['R'].iloc[-1]
final_R_det = sim_det['R'].iloc[-1]

# Peak I and time
peak_I_er = sim_er['I'].max()
Tpeak_I_er = sim_er['time'][sim_er['I'].idxmax()]
peak_I_ba = sim_ba['I'].max()
Tpeak_I_ba = sim_ba['time'][sim_ba['I'].idxmax()]
peak_I_det = sim_det['I'].max()
Tpeak_I_det = sim_det['time'][sim_det['I'].idxmax()]

# Epidemic duration: time when I drops below 1
try:
    duration_er = sim_er['time'][sim_er['I'] < 1].iloc[0]
except:
    duration_er = sim_er['time'].iloc[-1]
try:
    duration_ba = sim_ba['time'][sim_ba['I'] < 1].iloc[0]
except:
    duration_ba = sim_ba['time'].iloc[-1]
try:
    duration_det = sim_det['time'][sim_det['I'] < 1].iloc[0]
except:
    duration_det = sim_det['time'].iloc[-1]

# Collect all metrics as a dict and table
metrics_summary = {
    'Final_R': [final_R_er, final_R_ba, final_R_det],
    'Peak_I': [peak_I_er, peak_I_ba, peak_I_det],
    'Time_Peak_I': [Tpeak_I_er, Tpeak_I_ba, Tpeak_I_det],
    'Duration': [duration_er, duration_ba, duration_det]
}
results_table = pd.DataFrame(metrics_summary, index=['Stochastic_ER', 'Stochastic_BA', 'Deterministic(ODE)'])
results_table.to_csv('output/seir_metrics_table.csv')
print(results_table)
