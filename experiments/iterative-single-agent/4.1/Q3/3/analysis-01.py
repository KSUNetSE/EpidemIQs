
# Chain of Thought for Analysis:
# Now we read both simulation results: static (results-01.csv) and temporal (results-02.csv).
# We'll extract key metrics for each: epidemic peak (max I), final size (R at end), duration (when I drops below 1), peak time.
# We'll plot S/I/R curves for both, and produce a comparative summary table.

import pandas as pd
import matplotlib.pyplot as plt
import os

static_path = os.path.join(os.getcwd(), 'output', 'results-01.csv')
temporal_path = os.path.join(os.getcwd(), 'output', 'results-02.csv')

static = pd.read_csv(static_path)
temporal = pd.read_csv(temporal_path)

# For static: time, S, I, R columns, potentially many time points
times_static = static['time'].to_numpy()
S_static = static['S'].to_numpy()
I_static = static['I'].to_numpy()
R_static = static['R'].to_numpy()

# For temporal: S, I, R, time_step
S_t = temporal['S'].to_numpy()
I_t = temporal['I'].to_numpy()
R_t = temporal['R'].to_numpy()
times_t = temporal['time_step'].to_numpy()

# Metrics for static
peak_I_static = I_static.max()
peak_time_static = times_static[I_static.argmax()]
final_R_static = R_static[-1]
end_time_static = times_static[next(i for i,v in enumerate(I_static) if v<1)] if any(I_static<1) else times_static[-1]

# Metrics for temporal
peak_I_temporal = I_t.max()
peak_time_temporal = times_t[I_t.argmax()]
final_R_temporal = R_t[-1]
end_time_temporal = times_t[next(i for i,v in enumerate(I_t) if v<1)] if any(I_t<1) else times_t[-1]

# Plot comparisons
plt.figure(figsize=(10,5))
plt.plot(times_static, I_static, label='Infected (Static)')
plt.plot(times_t, I_t, label='Infected (Temporal)')
plt.xlabel('Time')
plt.ylabel('# Infected')
plt.legend()
plt.title('Comparison of Infected Fraction')
plt.savefig(os.path.join(os.getcwd(), 'output', 'comparison-infected.png'))
plt.close()

plt.figure(figsize=(10,5))
plt.plot(times_static, S_static, '--', label='Susceptible (Static)')
plt.plot(times_t, S_t, '--', label='Susceptible (Temporal)')
plt.plot(times_static, R_static, '-.', label='Recovered (Static)')
plt.plot(times_t, R_t, '-.', label='Recovered (Temporal)')
plt.xlabel('Time')
plt.ylabel('# Individuals')
plt.legend()
plt.title('S, I, R Curves: Static vs. Temporal')
plt.savefig(os.path.join(os.getcwd(), 'output', 'comparison-SIR.png'))
plt.close()

# Summary table
data = {
    'Metric': ['Peak Infected', 'Peak Time', 'Final Epidemic Size', 'Epidemic End Time'],
    'Static Network': [peak_I_static, peak_time_static, final_R_static, end_time_static],
    'Temporal Network': [peak_I_temporal, peak_time_temporal, final_R_temporal, end_time_temporal],
}
sumtab = pd.DataFrame(data)
sumtab.to_csv(os.path.join(os.getcwd(), 'output', 'summary-metrics.csv'), index=False)

return_vars = ['peak_I_static','peak_time_static','final_R_static','end_time_static','peak_I_temporal','peak_time_temporal','final_R_temporal','end_time_temporal']
