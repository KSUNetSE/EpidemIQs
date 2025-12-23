
import pandas as pd
import matplotlib.pyplot as plt
import os

df_static = pd.read_csv(os.path.join(os.getcwd(), 'output', 'results-10.csv'))
df_temporal = pd.read_csv(os.path.join(os.getcwd(), 'output', 'results-11.csv'))

plt.figure(figsize=(10,6))
plt.plot(df_static['time'], df_static['I'], label='Static Aggregated: Infected')
plt.plot(df_temporal['time'], df_temporal['I'], label='Temporal: Infected')
plt.plot(df_static['time'], df_static['S'], '--', label='Static Aggregated: Susceptible')
plt.plot(df_temporal['time'], df_temporal['S'], '--', label='Temporal: Susceptible')
plt.plot(df_static['time'], df_static['R'], ':', label='Static Aggregated: Recovered')
plt.plot(df_temporal['time'], df_temporal['R'], ':', label='Temporal: Recovered')
plt.xlabel('Time')
plt.ylabel('Average Number of Nodes')
plt.title('SIR Dynamics: Static Aggregated vs Activity-Driven Temporal Network')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(os.getcwd(), 'output', 'results-comparison.png'))
# Extract metrics
metrics = {}
def extract_metrics(df):
    peak_I = df['I'].max()
    peak_time = df['I'].idxmax()
    final_size = df['R'].iloc[-1]
    duration = df[df['I']>0].shape[0]
    return {'peak_infections': peak_I, 'peak_time': peak_time, 'final_epidemic_size': final_size, 'epidemic_duration': duration}
metrics['static'] = extract_metrics(df_static)
metrics['temporal'] = extract_metrics(df_temporal)
metrics_path = os.path.join(os.getcwd(), 'output', 'results-metrics.txt')
with open(metrics_path, 'w') as f:
    f.write(str(metrics))
metrics_path, os.path.join(os.getcwd(), 'output', 'results-comparison.png')