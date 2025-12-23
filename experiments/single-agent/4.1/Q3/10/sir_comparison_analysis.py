
import pandas as pd
import matplotlib.pyplot as plt
import os
# Static
static_df = pd.read_csv(os.path.join(os.getcwd(), 'output', 'results-11.csv'))
# Temporal
temporal_df = pd.read_csv(os.path.join(os.getcwd(), 'output', 'results-12.csv'))

# Plot both epidemics for comparison
plt.figure(figsize=(10, 5))
plt.plot(static_df['time'], static_df['I'], label='Static: Infected', color='red')
plt.plot(static_df['time'], static_df['S'], label='Static: Susceptible', color='blue')
plt.plot(static_df['time'], static_df['R'], label='Static: Recovered', color='green')
plt.plot(temporal_df['time'], temporal_df['I'], '--', label='Temporal: Infected', color='orange')
plt.plot(temporal_df['time'], temporal_df['S'], '--', label='Temporal: Susceptible', color='skyblue')
plt.plot(temporal_df['time'], temporal_df['R'], '--', label='Temporal: Recovered', color='lime')
plt.xlabel('Time steps')
plt.ylabel('Population count')
plt.title('SIR Epidemic: Static Aggregated vs. Activity-driven Temporal')
plt.legend()
plt.tight_layout()
fig_path = os.path.join(os.getcwd(), 'output', 'results-comparison.png')
plt.savefig(fig_path)
plt.close()

# Compute metrics: Peak I, time to peak, final size (R at t_end)
summary = {}
for label, df in [('static', static_df), ('temporal', temporal_df)]:
    peak_I = df['I'].max()
    t_peak = df['time'][df['I'].idxmax()]
    final_R = df['R'].iloc[-1]
    duration = df['time'][df['I'].to_numpy().nonzero()[0][-1]] if (df['I']>0).any() else 0
    summary[label] = {'peak_I': int(peak_I), 't_peak': float(t_peak), 'final_R': int(final_R), 'duration': float(duration)}
(fig_path, summary)